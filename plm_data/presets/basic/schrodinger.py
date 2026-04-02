"""Linear time-dependent Schrödinger equation via a real/imaginary split."""

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem

from plm_data.core.initial_conditions import apply_ic
from plm_data.core.spatial_fields import build_interpolator, scalar_expression_to_config
from plm_data.core.solver_strategies import CONSTANT_LHS_BLOCK_DIRECT
from plm_data.core.stochastic import build_scalar_coefficient
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, TransientLinearProblem
from plm_data.presets.boundary_validation import validate_boundary_field_structure
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    CoefficientSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

_SCHRODINGER_BOUNDARY_OPERATORS = {
    name: SCALAR_STANDARD_BOUNDARY_OPERATORS[name] for name in ("dirichlet", "periodic")
}

_SCHRODINGER_SPEC = PresetSpec(
    name="schrodinger",
    category="basic",
    description=(
        "Linear time-dependent Schrödinger equation solved via a real/imaginary "
        "split with a configurable external potential."
    ),
    equations={
        "u": "du/dt = D*lap(v) + V*v",
        "v": "dv/dt = -D*lap(u) - V*u",
    },
    parameters=[
        PDEParameter("D", "Dispersion coefficient"),
        PDEParameter("theta", "Time-stepping parameter (0.5 = Crank-Nicolson)"),
    ],
    inputs={
        "u": InputSpec(
            name="u",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "v": InputSpec(
            name="v",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "u": BoundaryFieldSpec(
            name="u",
            shape="scalar",
            operators=_SCHRODINGER_BOUNDARY_OPERATORS,
            description="Boundary conditions for Re(psi).",
        ),
        "v": BoundaryFieldSpec(
            name="v",
            shape="scalar",
            operators=_SCHRODINGER_BOUNDARY_OPERATORS,
            description="Boundary conditions for Im(psi).",
        ),
    },
    states={
        "u": StateSpec(name="u", shape="scalar"),
        "v": StateSpec(name="v", shape="scalar"),
    },
    outputs={
        "u": OutputSpec(
            name="u",
            shape="scalar",
            output_mode="scalar",
            source_name="u",
        ),
        "v": OutputSpec(
            name="v",
            shape="scalar",
            output_mode="scalar",
            source_name="v",
        ),
        "density": OutputSpec(
            name="density",
            shape="scalar",
            output_mode="scalar",
            source_name="density",
            source_kind="derived",
        ),
        "potential": OutputSpec(
            name="potential",
            shape="scalar",
            output_mode="scalar",
            source_name="potential",
            source_kind="derived",
        ),
    },
    static_fields=["potential"],
    steady_state=False,
    supported_dimensions=[2, 3],
    coefficients={
        "potential": CoefficientSpec(
            name="potential",
            shape="scalar",
            description="External potential field V(x).",
        )
    },
)


def _append_scalar_subspace_dirichlet_bcs(
    *,
    subspace,
    domain_geom,
    boundary_field,
    parameters: dict[str, float],
) -> list[fem.DirichletBC]:
    """Build scalar Dirichlet BCs on one mixed scalar subspace."""
    fdim = domain_geom.mesh.topology.dim - 1
    collapsed_space, _ = subspace.collapse()
    bcs: list[fem.DirichletBC] = []

    for side_name, entries in boundary_field.sides.items():
        dofs = None
        for bc in entries:
            if bc.type != "dirichlet":
                continue
            if bc.value is None:
                raise ValueError(f"Dirichlet BC on '{side_name}' requires a value")
            if bc.value.is_componentwise:
                raise ValueError(
                    "Scalar Dirichlet BCs cannot use component-wise values"
                )

            if dofs is None:
                facets = domain_geom.facet_tags.find(
                    domain_geom.boundary_names[side_name]
                )
                dofs = fem.locate_dofs_topological(
                    (subspace, collapsed_space),
                    fdim,
                    facets,
                )

            interp = build_interpolator(
                scalar_expression_to_config(bc.value),
                parameters,
            )
            if interp is None:
                raise ValueError(
                    f"Dirichlet BC on '{side_name}' cannot use custom values"
                )

            bc_func = fem.Function(collapsed_space)
            bc_func.interpolate(interp)
            bcs.append(fem.dirichletbc(bc_func, dofs, subspace))

    return bcs


def _build_split_scalar_dirichlet_bcs(
    mixed_space: fem.FunctionSpace,
    domain_geom,
    boundary_field_u,
    boundary_field_v,
    parameters: dict[str, float],
) -> list[fem.DirichletBC]:
    """Build Dirichlet BCs for the real and imaginary subspaces."""
    return _append_scalar_subspace_dirichlet_bcs(
        subspace=mixed_space.sub(0),
        domain_geom=domain_geom,
        boundary_field=boundary_field_u,
        parameters=parameters,
    ) + _append_scalar_subspace_dirichlet_bcs(
        subspace=mixed_space.sub(1),
        domain_geom=domain_geom,
        boundary_field=boundary_field_v,
        parameters=parameters,
    )


class _SchrodingerProblem(TransientLinearProblem):
    supported_solver_strategies = (CONSTANT_LHS_BLOCK_DIRECT,)

    def validate_boundary_conditions(self, domain_geom):
        boundary_field_u = self.config.boundary_field("u")
        boundary_field_v = self.config.boundary_field("v")

        validate_boundary_field_structure(
            preset_name=self.spec.name,
            field_name="u",
            boundary_field=boundary_field_u,
            domain_geom=domain_geom,
            allowed_operators={"dirichlet", "periodic"},
        )
        validate_boundary_field_structure(
            preset_name=self.spec.name,
            field_name="v",
            boundary_field=boundary_field_v,
            domain_geom=domain_geom,
            allowed_operators={"dirichlet", "periodic"},
        )

        if (
            boundary_field_u.periodic_pair_keys()
            != boundary_field_v.periodic_pair_keys()
        ):
            raise ValueError(
                "Schrodinger real and imaginary boundary conditions must use "
                "identical periodic side pairs."
            )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        params = self.config.parameters
        D = params["D"]
        theta = params["theta"]
        dt = self.config.time.dt

        scalar_element = element(
            "Lagrange",
            self.msh.basix_cell(),
            1,
            dtype=default_real_type,
        )
        mixed_space = fem.functionspace(
            self.msh,
            mixed_element([scalar_element, scalar_element]),
        )

        boundary_field_u = self.config.boundary_field("u")
        boundary_field_v = self.config.boundary_field("v")
        bcs = _build_split_scalar_dirichlet_bcs(
            mixed_space,
            domain_geom,
            boundary_field_u,
            boundary_field_v,
            params,
        )
        mpc = self.create_periodic_constraint(
            mixed_space,
            domain_geom,
            boundary_field_u,
            bcs,
            constrained_spaces=[mixed_space.sub(0), mixed_space.sub(1)],
        )
        solution_space = mixed_space if mpc is None else mpc.function_space

        self.w = fem.Function(solution_space)
        self.w0 = fem.Function(solution_space)

        input_u = self.config.input("u")
        input_v = self.config.input("v")
        assert input_u.initial_condition is not None
        assert input_v.initial_condition is not None
        apply_ic(
            self.w.sub(0), input_u.initial_condition, params, seed=self.config.seed
        )
        apply_ic(
            self.w.sub(1), input_v.initial_condition, params, seed=self.config.seed
        )
        for bc in bcs:
            bc.set(self.w.x.array)
        self.w.x.scatter_forward()
        self.w0.x.array[:] = self.w.x.array
        for bc in bcs:
            bc.set(self.w0.x.array)
        self.w0.x.scatter_forward()

        u, v = ufl.TrialFunctions(mixed_space)
        u0, v0 = ufl.split(self.w0)
        phi_u, phi_v = ufl.TestFunctions(mixed_space)
        dt_c = fem.Constant(self.msh, default_real_type(dt))

        potential = build_scalar_coefficient(self, "potential")
        if potential is None:
            raise ValueError(
                "Schrodinger coefficient 'potential' cannot use a custom expression"
            )

        operator_uv = (
            D * ufl.inner(ufl.grad(v), ufl.grad(phi_u)) * ufl.dx
            + potential * v * phi_u * ufl.dx
        )
        operator_vu = (
            D * ufl.inner(ufl.grad(u), ufl.grad(phi_v)) * ufl.dx
            + potential * u * phi_v * ufl.dx
        )
        rhs_operator_uv = (
            D * ufl.inner(ufl.grad(v0), ufl.grad(phi_u)) * ufl.dx
            + potential * v0 * phi_u * ufl.dx
        )
        rhs_operator_vu = (
            D * ufl.inner(ufl.grad(u0), ufl.grad(phi_v)) * ufl.dx
            + potential * u0 * phi_v * ufl.dx
        )

        a = (
            ufl.inner(u, phi_u) * ufl.dx
            - theta * dt_c * operator_uv
            + ufl.inner(v, phi_v) * ufl.dx
            + theta * dt_c * operator_vu
        )
        L = (
            ufl.inner(u0, phi_u) * ufl.dx
            + (1.0 - theta) * dt_c * rhs_operator_uv
            + ufl.inner(v0, phi_v) * ufl.dx
            - (1.0 - theta) * dt_c * rhs_operator_vu
        )

        self.problem = self.create_linear_problem(
            a,
            L,
            u=self.w,
            bcs=bcs,
            petsc_options_prefix="plm_schrodinger_",
            mpc=mpc,
        )

        V_u, self._u_dofs = self.w.function_space.sub(0).collapse()
        self.u_out = fem.Function(V_u, name="u")
        V_v, self._v_dofs = self.w.function_space.sub(1).collapse()
        self.v_out = fem.Function(V_v, name="v")
        self.density_out = fem.Function(V_u, name="density")
        self.potential_out = fem.Function(V_u, name="potential")

        potential_interp = build_interpolator(
            scalar_expression_to_config(self.config.coefficient("potential")),
            params,
        )
        if potential_interp is None:
            raise ValueError(
                "Schrodinger coefficient 'potential' cannot use a custom expression"
            )
        self.potential_out.interpolate(potential_interp)
        self.potential_out.x.scatter_forward()

    def step(self, t: float, dt: float) -> bool:
        self.w0.x.array[:] = self.w.x.array
        self.w0.x.scatter_forward()
        self.problem.solve()
        return self.problem.solver.getConvergedReason() > 0

    def get_output_fields(self) -> dict[str, fem.Function]:
        u_values = self.w.x.array[self._u_dofs]
        v_values = self.w.x.array[self._v_dofs]
        self.u_out.x.array[:] = u_values
        self.v_out.x.array[:] = v_values
        self.density_out.x.array[:] = np.square(u_values) + np.square(v_values)
        return {
            "u": self.u_out,
            "v": self.v_out,
            "density": self.density_out,
            "potential": self.potential_out,
        }

    def get_num_dofs(self) -> int:
        return self.w.function_space.dofmap.index_map.size_global


@register_preset("schrodinger")
class SchrodingerPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _SCHRODINGER_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _SchrodingerProblem(self.spec, config)

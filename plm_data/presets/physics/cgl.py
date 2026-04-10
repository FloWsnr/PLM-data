"""Complex Ginzburg-Landau equation preset.

Solves the general-coefficient CGL/NLS equation via a real/imaginary split:
    dA/dt = (D_r + i*D_i) laplacian(A) + (a_r + i*a_i) A + (b_r + i*b_i) |A|^2 A

Writing A = u + iv gives the coupled real system:
    du/dt = D_r*lap(u) - D_i*lap(v) + a_r*u - a_i*v + |A|^2*(b_r*u - b_i*v)
    dv/dt = D_i*lap(u) + D_r*lap(v) + a_i*u + a_r*v + |A|^2*(b_i*u + b_r*v)

Canonical CGL: D_r=1, D_i=b, a_r=r, a_i=0, b_r=-1, b_i=-c.
Benjamin-Feir instability (chaos) when 1 + b*c < 0.
"""

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem

from plm_data.core.boundary_conditions import apply_dirichlet_bcs_to_subspace
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.solver_strategies import NONLINEAR_MIXED_DIRECT
from plm_data.core.spatial_fields import is_exact_zero_field_expression
from plm_data.presets import register_preset
from plm_data.presets.base import (
    PDEPreset,
    ProblemInstance,
    TransientNonlinearProblem,
)
from plm_data.presets.boundary_validation import validate_boundary_field_structure
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

_CGL_BOUNDARY_OPERATORS = {
    name: SCALAR_STANDARD_BOUNDARY_OPERATORS[name]
    for name in ("dirichlet", "neumann", "periodic")
}

_CGL_SPEC = PresetSpec(
    name="cgl",
    category="physics",
    description=(
        "Complex Ginzburg-Landau equation solved via real/imaginary split. "
        "Covers canonical CGL (spatiotemporal chaos) and NLS (solitons) as "
        "special cases of the general-coefficient form."
    ),
    equations={
        "u": "du/dt = D_r*lap(u) - D_i*lap(v) + a_r*u - a_i*v + |A|^2*(b_r*u - b_i*v)",
        "v": "dv/dt = D_i*lap(u) + D_r*lap(v) + a_i*u + a_r*v + |A|^2*(b_i*u + b_r*v)",
    },
    parameters=[
        PDEParameter("D_r", "Real part of diffusion coefficient"),
        PDEParameter("D_i", "Imaginary part of diffusion coefficient (dispersion)"),
        PDEParameter("a_r", "Real part of linear growth/decay coefficient"),
        PDEParameter("a_i", "Imaginary part of linear coefficient"),
        PDEParameter("b_r", "Real part of nonlinear saturation coefficient"),
        PDEParameter("b_i", "Imaginary part of nonlinear dispersion coefficient"),
        PDEParameter("theta", "Time-stepping parameter (0.5=Crank-Nicolson)"),
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
            operators=_CGL_BOUNDARY_OPERATORS,
            description="Boundary conditions for Re(A).",
        ),
        "v": BoundaryFieldSpec(
            name="v",
            shape="scalar",
            operators=_CGL_BOUNDARY_OPERATORS,
            description="Boundary conditions for Im(A).",
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
        "amplitude": OutputSpec(
            name="amplitude",
            shape="scalar",
            output_mode="scalar",
            source_name="amplitude",
            source_kind="derived",
        ),
    },
    static_fields=[],
    steady_state=False,
    supported_dimensions=[2, 3],
)


class _CGLProblem(TransientNonlinearProblem):
    supported_solver_strategies = (NONLINEAR_MIXED_DIRECT,)

    def validate_boundary_conditions(self, domain_geom):
        boundary_field_u = self.config.boundary_field("u")
        boundary_field_v = self.config.boundary_field("v")
        for field_name in ("u", "v"):
            boundary_field = self.config.boundary_field(field_name)
            validate_boundary_field_structure(
                preset_name=self.spec.name,
                field_name=field_name,
                boundary_field=boundary_field,
                domain_geom=domain_geom,
                allowed_operators=set(_CGL_BOUNDARY_OPERATORS),
            )
            self._validate_homogeneous_neumann(field_name, boundary_field)

        if (
            boundary_field_u.periodic_pair_keys()
            != boundary_field_v.periodic_pair_keys()
        ):
            raise ValueError(
                "CGL requires matching periodic side pairs for the real and imaginary "
                "fields because both components share one mixed periodic constraint."
            )

    def _validate_homogeneous_neumann(self, field_name, boundary_field) -> None:
        for side_name, entries in boundary_field.sides.items():
            for entry in entries:
                if entry.type != "neumann":
                    continue
                if entry.value is None:
                    raise ValueError(
                        f"CGL boundary field '{field_name}' side '{side_name}' "
                        "uses Neumann but is missing a value."
                    )
                if not is_exact_zero_field_expression(
                    entry.value,
                    self.config.parameters,
                ):
                    raise ValueError(
                        "CGL currently supports homogeneous Neumann boundaries only. "
                        f"Field '{field_name}' side '{side_name}' has a nonzero "
                        "Neumann value."
                    )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        boundary_field_u = self.config.boundary_field("u")
        boundary_field_v = self.config.boundary_field("v")

        P1 = element("Lagrange", self.msh.basix_cell(), 1, dtype=default_real_type)
        ME = fem.functionspace(self.msh, mixed_element([P1, P1]))

        params = self.config.parameters
        D_r = params["D_r"]
        D_i = params["D_i"]
        a_r = params["a_r"]
        a_i = params["a_i"]
        b_r = params["b_r"]
        b_i = params["b_i"]
        theta = params["theta"]
        dt = self.config.time.dt

        # Periodic constraint on both subspaces (u and v share the same BC structure)
        bcs = [
            *apply_dirichlet_bcs_to_subspace(
                ME.sub(0),
                domain_geom,
                boundary_field_u,
                params,
            ),
            *apply_dirichlet_bcs_to_subspace(
                ME.sub(1),
                domain_geom,
                boundary_field_v,
                params,
            ),
        ]
        mpc = self.create_periodic_constraint(
            ME,
            domain_geom,
            boundary_field_u,
            bcs,
            constrained_spaces=[ME.sub(0), ME.sub(1)],
        )
        solution_space = ME if mpc is None else mpc.function_space

        self.w = fem.Function(solution_space)
        self.w0 = fem.Function(solution_space)

        # Apply initial conditions to each component
        ic_u = self.config.input("u").initial_condition
        ic_v = self.config.input("v").initial_condition
        assert ic_u is not None
        assert ic_v is not None
        apply_ic(self.w.sub(0), ic_u, params, seed=self.config.seed)
        # Use a different seed for v to get independent noise
        seed_v = self.config.seed + 1 if self.config.seed is not None else None
        apply_ic(self.w.sub(1), ic_v, params, seed=seed_v)
        self.w.x.scatter_forward()
        self.w0.x.array[:] = self.w.x.array

        # Split into components
        u, v = ufl.split(self.w)
        u0, v0 = ufl.split(self.w0)
        phi_u, phi_v = ufl.TestFunctions(ME)

        # Theta-method interpolation
        u_mid = (1.0 - theta) * u0 + theta * u
        v_mid = (1.0 - theta) * v0 + theta * v

        # |A|^2 at old time level (semi-implicit linearization)
        A_sq = u0**2 + v0**2

        dt_c = fem.Constant(self.msh, default_real_type(dt))

        # Real part residual:
        #   (u - u0)/dt + D_r*grad(u_mid).grad(phi_u) - D_i*grad(v_mid).grad(phi_u)
        #   - (a_r*u_mid - a_i*v_mid)*phi_u - A_sq*(b_r*u_mid - b_i*v_mid)*phi_u = 0
        F_u = (
            ufl.inner(u - u0, phi_u) * ufl.dx
            + dt_c * D_r * ufl.inner(ufl.grad(u_mid), ufl.grad(phi_u)) * ufl.dx
            - dt_c * D_i * ufl.inner(ufl.grad(v_mid), ufl.grad(phi_u)) * ufl.dx
            - dt_c * (a_r * u_mid - a_i * v_mid) * phi_u * ufl.dx
            - dt_c * A_sq * (b_r * u_mid - b_i * v_mid) * phi_u * ufl.dx
        )

        # Imaginary part residual:
        #   (v - v0)/dt + D_i*grad(u_mid).grad(phi_v) + D_r*grad(v_mid).grad(phi_v)
        #   - (a_i*u_mid + a_r*v_mid)*phi_v - A_sq*(b_i*u_mid + b_r*v_mid)*phi_v = 0
        F_v = (
            ufl.inner(v - v0, phi_v) * ufl.dx
            + dt_c * D_i * ufl.inner(ufl.grad(u_mid), ufl.grad(phi_v)) * ufl.dx
            + dt_c * D_r * ufl.inner(ufl.grad(v_mid), ufl.grad(phi_v)) * ufl.dx
            - dt_c * (a_i * u_mid + a_r * v_mid) * phi_v * ufl.dx
            - dt_c * A_sq * (b_i * u_mid + b_r * v_mid) * phi_v * ufl.dx
        )

        F = F_u + F_v
        J = ufl.derivative(F, self.w, ufl.TrialFunction(ME))

        self.problem = self.create_nonlinear_problem(
            F,
            self.w,
            bcs=bcs,
            petsc_options_prefix="plm_cgl_",
            J=J,
            mpc=mpc,
        )

        # Collapsed output functions
        V0, self._u_dofs = self.w.function_space.sub(0).collapse()
        self.u_out = fem.Function(V0, name="u")
        V1, self._v_dofs = self.w.function_space.sub(1).collapse()
        self.v_out = fem.Function(V1, name="v")
        self.amplitude_out = fem.Function(V0, name="amplitude")

    def step(self, t: float, dt: float) -> bool:
        self.w0.x.array[:] = self.w.x.array
        self.problem.solve()
        return self.problem.solver.getConvergedReason() > 0

    def get_output_fields(self) -> dict[str, fem.Function]:
        u_arr = self.w.x.array[self._u_dofs]
        v_arr = self.w.x.array[self._v_dofs]
        self.u_out.x.array[:] = u_arr
        self.v_out.x.array[:] = v_arr
        self.amplitude_out.x.array[:] = np.sqrt(u_arr**2 + v_arr**2)
        return {"u": self.u_out, "v": self.v_out, "amplitude": self.amplitude_out}

    def get_num_dofs(self) -> int:
        return self.w.function_space.dofmap.index_map.size_global


@register_preset("cgl")
class CGLPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _CGL_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _CGLProblem(self.spec, config)

"""Gray-Scott reaction-diffusion preset."""

import numpy as np
import ufl
from dolfinx import fem

from plm_data.core.boundary_conditions import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.solver_strategies import CONSTANT_LHS_SCALAR_SPD
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, TransientLinearProblem
from plm_data.presets.boundary_validation import (
    validate_scalar_standard_boundary_field,
)
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

_GRAY_SCOTT_SPEC = PresetSpec(
    name="gray_scott",
    category="physics",
    description=(
        "Gray-Scott reaction-diffusion system for substrate/autocatalyst "
        "pattern formation."
    ),
    equations={
        "u": "du/dt = Du * laplacian(u) - u * v^2 + F * (1 - u)",
        "v": "dv/dt = Dv * laplacian(v) + u * v^2 - (F + k) * v",
    },
    parameters=[
        PDEParameter("Du", "Diffusion coefficient of the substrate field u"),
        PDEParameter("Dv", "Diffusion coefficient of the autocatalyst field v"),
        PDEParameter("F", "Feed rate"),
        PDEParameter("k", "Kill rate"),
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
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the substrate field.",
        ),
        "v": BoundaryFieldSpec(
            name="v",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the autocatalyst field.",
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
    },
    static_fields=[],
    steady_state=False,
    supported_dimensions=[2, 3],
)


def _space_num_dofs(V: fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


def _gray_scott_patch_interpolator(*, ic_config, gdim: int):
    center = ic_config.params["center"]
    half_width = ic_config.params["half_width"]
    background = float(ic_config.params["background"])
    patch_value = float(ic_config.params["patch_value"])

    if len(center) != gdim:
        raise ValueError(
            f"gray_scott_patch center must have {gdim} entries in {gdim}D. "
            f"Got {center!r}."
        )
    if len(half_width) != gdim:
        raise ValueError(
            f"gray_scott_patch half_width must have {gdim} entries in {gdim}D. "
            f"Got {half_width!r}."
        )

    center_array = np.asarray(center, dtype=float)
    half_width_array = np.asarray(half_width, dtype=float)

    def _interpolator(x: np.ndarray) -> np.ndarray:
        inside_patch = np.ones(x.shape[1], dtype=bool)
        for axis in range(gdim):
            inside_patch &= (
                np.abs(x[axis] - center_array[axis]) <= half_width_array[axis]
            )
        return np.where(inside_patch, patch_value, background)

    return _interpolator


def _apply_gray_scott_ic(
    func: fem.Function,
    *,
    ic_config,
    parameters: dict[str, float],
    seed: int | None,
) -> None:
    if ic_config.type == "gray_scott_patch":
        func.interpolate(
            _gray_scott_patch_interpolator(
                ic_config=ic_config,
                gdim=func.function_space.mesh.geometry.dim,
            )
        )
        return

    apply_ic(func, ic_config, parameters, seed=seed)


class _GrayScottProblem(TransientLinearProblem):
    supported_solver_strategies = (CONSTANT_LHS_SCALAR_SPD,)

    def validate_boundary_conditions(self, domain_geom):
        validate_scalar_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="u",
            boundary_field=self.config.boundary_field("u"),
            domain_geom=domain_geom,
        )
        validate_scalar_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="v",
            boundary_field=self.config.boundary_field("v"),
            domain_geom=domain_geom,
        )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        V = fem.functionspace(self.msh, ("Lagrange", 1))

        u_input = self.config.input("u")
        v_input = self.config.input("v")
        u_boundary_field = self.config.boundary_field("u")
        v_boundary_field = self.config.boundary_field("v")

        dt = self.config.time.dt
        Du = self.config.parameters["Du"]
        Dv = self.config.parameters["Dv"]
        F = self.config.parameters["F"]
        k = self.config.parameters["k"]

        u_bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            u_boundary_field,
            self.config.parameters,
        )
        v_bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            v_boundary_field,
            self.config.parameters,
        )
        u_mpc = self.create_periodic_constraint(
            V,
            domain_geom,
            u_boundary_field,
            u_bcs,
        )
        v_mpc = self.create_periodic_constraint(
            V,
            domain_geom,
            v_boundary_field,
            v_bcs,
        )

        self.V_u = V if u_mpc is None else u_mpc.function_space
        self.V_v = V if v_mpc is None else v_mpc.function_space
        self._num_dofs = _space_num_dofs(self.V_u) + _space_num_dofs(self.V_v)

        self.u_n = fem.Function(self.V_u, name="u")
        self.v_n = fem.Function(self.V_v, name="v")
        self.u_h = fem.Function(self.V_u, name="u_next")
        self.v_h = fem.Function(self.V_v, name="v_next")

        assert u_input.initial_condition is not None
        _apply_gray_scott_ic(
            self.u_n,
            ic_config=u_input.initial_condition,
            parameters=self.config.parameters,
            seed=self.config.seed,
        )
        assert v_input.initial_condition is not None
        _apply_gray_scott_ic(
            self.v_n,
            ic_config=v_input.initial_condition,
            parameters=self.config.parameters,
            seed=self.config.seed,
        )
        self.u_n.x.scatter_forward()
        self.v_n.x.scatter_forward()

        u_trial = ufl.TrialFunction(V)
        u_test = ufl.TestFunction(V)
        v_trial = ufl.TrialFunction(V)
        v_test = ufl.TestFunction(V)

        cubic_reaction = self.u_n * self.v_n**2

        a_u = (
            ufl.inner(u_trial, u_test) * ufl.dx
            + dt * Du * ufl.inner(ufl.grad(u_trial), ufl.grad(u_test)) * ufl.dx
            + dt * F * ufl.inner(u_trial, u_test) * ufl.dx
        )
        L_u = (
            ufl.inner(self.u_n, u_test) * ufl.dx
            + dt * ufl.inner(F - cubic_reaction, u_test) * ufl.dx
        )

        a_v = (
            ufl.inner(v_trial, v_test) * ufl.dx
            + dt * Dv * ufl.inner(ufl.grad(v_trial), ufl.grad(v_test)) * ufl.dx
            + dt * (F + k) * ufl.inner(v_trial, v_test) * ufl.dx
        )
        L_v = (
            ufl.inner(self.v_n, v_test) * ufl.dx
            + dt * ufl.inner(cubic_reaction, v_test) * ufl.dx
        )

        a_u_bc, L_u_bc = build_natural_bc_forms(
            u_trial,
            u_test,
            domain_geom,
            u_boundary_field,
            self.config.parameters,
        )
        if a_u_bc is not None:
            a_u = a_u + dt * a_u_bc
        if L_u_bc is not None:
            L_u = L_u + dt * L_u_bc

        a_v_bc, L_v_bc = build_natural_bc_forms(
            v_trial,
            v_test,
            domain_geom,
            v_boundary_field,
            self.config.parameters,
        )
        if a_v_bc is not None:
            a_v = a_v + dt * a_v_bc
        if L_v_bc is not None:
            L_v = L_v + dt * L_v_bc

        self._u_problem = self.create_linear_problem(
            a_u,
            L_u,
            u=self.u_h,
            bcs=u_bcs,
            petsc_options_prefix="plm_gray_scott_u_",
            mpc=u_mpc,
        )
        self._v_problem = self.create_linear_problem(
            a_v,
            L_v,
            u=self.v_h,
            bcs=v_bcs,
            petsc_options_prefix="plm_gray_scott_v_",
            mpc=v_mpc,
        )

    def step(self, t: float, dt: float) -> bool:
        self._u_problem.solve()
        self._v_problem.solve()

        self.u_n.x.array[:] = self.u_h.x.array
        self.u_n.x.scatter_forward()
        self.v_n.x.array[:] = self.v_h.x.array
        self.v_n.x.scatter_forward()

        return (
            self._u_problem.solver.getConvergedReason() > 0
            and self._v_problem.solver.getConvergedReason() > 0
        )

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"u": self.u_n, "v": self.v_n}

    def get_num_dofs(self) -> int:
        return self._num_dofs


@register_preset("gray_scott")
class GrayScottPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _GRAY_SCOTT_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _GrayScottProblem(self.spec, config)

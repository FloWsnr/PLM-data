"""FitzHugh-Nagumo reaction-diffusion preset."""

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

_FITZHUGH_NAGUMO_SPEC = PresetSpec(
    name="fitzhugh_nagumo",
    category="biology",
    description=(
        "FitzHugh-Nagumo reaction-diffusion system for neuronal excitability. "
        "Two-species activator-inhibitor model supporting Turing patterns, "
        "spiral waves, and excitable dynamics. "
        "Turing instability requires Dv/Du >> 1 with small tau."
    ),
    equations={
        "u": "du/dt = Du * laplacian(u) + u - u^3 - v",
        "v": "tau * dv/dt = Dv * laplacian(v) + u - b*v + a",
    },
    parameters=[
        PDEParameter("Du", "Diffusion coefficient of activator u"),
        PDEParameter("Dv", "Diffusion coefficient of inhibitor v"),
        PDEParameter("tau", "Inhibitor timescale ratio"),
        PDEParameter("b", "Recovery sensitivity to activator"),
        PDEParameter("a", "Threshold/offset parameter"),
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
            description="Boundary conditions for activator u.",
        ),
        "v": BoundaryFieldSpec(
            name="v",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for inhibitor v.",
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


class _FitzHughNagumoProblem(TransientLinearProblem):
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
        tau = self.config.parameters["tau"]
        b = self.config.parameters["b"]
        a = self.config.parameters["a"]

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
        apply_ic(
            self.u_n,
            u_input.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        if u_mpc is not None:
            u_mpc.backsubstitution(self.u_n)
        assert v_input.initial_condition is not None
        apply_ic(
            self.v_n,
            v_input.initial_condition,
            self.config.parameters,
            seed=(self.config.seed + 1) if self.config.seed is not None else None,
        )
        if v_mpc is not None:
            v_mpc.backsubstitution(self.v_n)
        self.u_n.x.scatter_forward()
        self.v_n.x.scatter_forward()

        u_trial = ufl.TrialFunction(V)
        u_test = ufl.TestFunction(V)
        v_trial = ufl.TrialFunction(V)
        v_test = ufl.TestFunction(V)

        # u-equation: (u_h - u_n)/dt = Du*laplacian(u_h) + u_n - u_n^3 - v_n
        # All reaction terms explicit (the +u term is destabilizing, so
        # implicit treatment would break SPD-ness of the LHS)
        lhs_u = (
            ufl.inner(u_trial, u_test) * ufl.dx
            + dt * Du * ufl.inner(ufl.grad(u_trial), ufl.grad(u_test)) * ufl.dx
        )
        rhs_u = (
            ufl.inner(self.u_n, u_test) * ufl.dx
            + dt * ufl.inner(self.u_n - self.u_n**3 - self.v_n, u_test) * ufl.dx
        )

        # v-equation: tau*(v_h - v_n)/dt = Dv*laplacian(v_h) - b*v_h + u_n + a
        # Linear decay (-b*v) treated implicitly; tau scales the time derivative
        lhs_v = (
            tau * ufl.inner(v_trial, v_test) * ufl.dx
            + dt * Dv * ufl.inner(ufl.grad(v_trial), ufl.grad(v_test)) * ufl.dx
            + dt * b * ufl.inner(v_trial, v_test) * ufl.dx
        )
        rhs_v = (
            tau * ufl.inner(self.v_n, v_test) * ufl.dx
            + dt * ufl.inner(self.u_n + a, v_test) * ufl.dx
        )

        lhs_u_bc, rhs_u_bc = build_natural_bc_forms(
            u_trial,
            u_test,
            domain_geom,
            u_boundary_field,
            self.config.parameters,
        )
        if lhs_u_bc is not None:
            lhs_u = lhs_u + dt * lhs_u_bc
        if rhs_u_bc is not None:
            rhs_u = rhs_u + dt * rhs_u_bc

        lhs_v_bc, rhs_v_bc = build_natural_bc_forms(
            v_trial,
            v_test,
            domain_geom,
            v_boundary_field,
            self.config.parameters,
        )
        if lhs_v_bc is not None:
            lhs_v = lhs_v + dt * lhs_v_bc
        if rhs_v_bc is not None:
            rhs_v = rhs_v + dt * rhs_v_bc

        self._u_problem = self.create_linear_problem(
            lhs_u,
            rhs_u,
            u=self.u_h,
            bcs=u_bcs,
            petsc_options_prefix="plm_fitzhugh_nagumo_u_",
            mpc=u_mpc,
        )
        self._v_problem = self.create_linear_problem(
            lhs_v,
            rhs_v,
            u=self.v_h,
            bcs=v_bcs,
            petsc_options_prefix="plm_fitzhugh_nagumo_v_",
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


@register_preset("fitzhugh_nagumo")
class FitzHughNagumoPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _FITZHUGH_NAGUMO_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _FitzHughNagumoProblem(self.spec, config)

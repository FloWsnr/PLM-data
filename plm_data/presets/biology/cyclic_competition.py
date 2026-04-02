"""Cyclic competition (rock-paper-scissors) reaction-diffusion preset."""

import ufl
from dolfinx import fem

from plm_data.core.boundary_conditions import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.stochastic import build_scalar_state_stochastic_term
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
    SATURATING_STOCHASTIC_COUPLINGS,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

_CYCLIC_COMPETITION_SPEC = PresetSpec(
    name="cyclic_competition",
    category="biology",
    description=(
        "Three-species competitive Lotka-Volterra system with cyclic dominance "
        "(rock-paper-scissors). Produces spiral waves and spatiotemporal chaos "
        "with unequal diffusion coefficients."
    ),
    equations={
        "u": "du/dt = Du * laplacian(u) + u*(1 - u - a*v - b*w)",
        "v": "dv/dt = Dv * laplacian(v) + v*(1 - b*u - v - a*w)",
        "w": "dw/dt = Dw * laplacian(w) + w*(1 - a*u - b*v - w)",
    },
    parameters=[
        PDEParameter("Du", "Diffusion coefficient of species u"),
        PDEParameter("Dv", "Diffusion coefficient of species v"),
        PDEParameter("Dw", "Diffusion coefficient of species w"),
        PDEParameter("a", "Weak interspecific competition coefficient (0 < a < 1)"),
        PDEParameter("b", "Strong interspecific competition coefficient (b > 1)"),
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
        "w": InputSpec(
            name="w",
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
            description="Boundary conditions for species u.",
        ),
        "v": BoundaryFieldSpec(
            name="v",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for species v.",
        ),
        "w": BoundaryFieldSpec(
            name="w",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for species w.",
        ),
    },
    states={
        "u": StateSpec(
            name="u",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
        "v": StateSpec(
            name="v",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
        "w": StateSpec(
            name="w",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
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
        "w": OutputSpec(
            name="w",
            shape="scalar",
            output_mode="scalar",
            source_name="w",
        ),
    },
    static_fields=[],
    steady_state=False,
    supported_dimensions=[2, 3],
)


def _space_num_dofs(V: fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


class _CyclicCompetitionProblem(TransientLinearProblem):
    supported_solver_strategies = (CONSTANT_LHS_SCALAR_SPD,)

    def validate_boundary_conditions(self, domain_geom):
        for field_name in ("u", "v", "w"):
            validate_scalar_standard_boundary_field(
                preset_name=self.spec.name,
                field_name=field_name,
                boundary_field=self.config.boundary_field(field_name),
                domain_geom=domain_geom,
            )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        V = fem.functionspace(self.msh, ("Lagrange", 1))

        dt = self.config.time.dt
        params = self.config.parameters
        Du = params["Du"]
        Dv = params["Dv"]
        Dw = params["Dw"]
        a = params["a"]
        b = params["b"]

        u_boundary_field = self.config.boundary_field("u")
        v_boundary_field = self.config.boundary_field("v")
        w_boundary_field = self.config.boundary_field("w")

        u_bcs = apply_dirichlet_bcs(V, domain_geom, u_boundary_field, params)
        v_bcs = apply_dirichlet_bcs(V, domain_geom, v_boundary_field, params)
        w_bcs = apply_dirichlet_bcs(V, domain_geom, w_boundary_field, params)

        u_mpc = self.create_periodic_constraint(V, domain_geom, u_boundary_field, u_bcs)
        v_mpc = self.create_periodic_constraint(V, domain_geom, v_boundary_field, v_bcs)
        w_mpc = self.create_periodic_constraint(V, domain_geom, w_boundary_field, w_bcs)

        self.V_u = V if u_mpc is None else u_mpc.function_space
        self.V_v = V if v_mpc is None else v_mpc.function_space
        self.V_w = V if w_mpc is None else w_mpc.function_space
        self._num_dofs = (
            _space_num_dofs(self.V_u)
            + _space_num_dofs(self.V_v)
            + _space_num_dofs(self.V_w)
        )

        self.u_n = fem.Function(self.V_u, name="u")
        self.v_n = fem.Function(self.V_v, name="v")
        self.w_n = fem.Function(self.V_w, name="w")
        self.u_h = fem.Function(self.V_u, name="u_next")
        self.v_h = fem.Function(self.V_v, name="v_next")
        self.w_h = fem.Function(self.V_w, name="w_next")

        u_ic = self.config.input("u").initial_condition
        v_ic = self.config.input("v").initial_condition
        w_ic = self.config.input("w").initial_condition
        assert u_ic is not None
        assert v_ic is not None
        assert w_ic is not None

        apply_ic(self.u_n, u_ic, params, seed=self.config.seed)
        if u_mpc is not None:
            u_mpc.backsubstitution(self.u_n)
        seed_v = (self.config.seed + 1) if self.config.seed is not None else None
        apply_ic(self.v_n, v_ic, params, seed=seed_v)
        if v_mpc is not None:
            v_mpc.backsubstitution(self.v_n)
        seed_w = (self.config.seed + 2) if self.config.seed is not None else None
        apply_ic(self.w_n, w_ic, params, seed=seed_w)
        if w_mpc is not None:
            w_mpc.backsubstitution(self.w_n)
        self.u_n.x.scatter_forward()
        self.v_n.x.scatter_forward()
        self.w_n.x.scatter_forward()

        u_trial = ufl.TrialFunction(V)
        u_test = ufl.TestFunction(V)
        v_trial = ufl.TrialFunction(V)
        v_test = ufl.TestFunction(V)
        w_trial = ufl.TrialFunction(V)
        w_test = ufl.TestFunction(V)

        # Fully explicit reactions, implicit diffusion.
        # u-equation: (u_h - u_n)/dt = Du*laplacian(u_h) + u_n*(1 - u_n - a*v_n - b*w_n)
        reaction_u = self.u_n * (1 - self.u_n - a * self.v_n - b * self.w_n)
        a_u = (
            ufl.inner(u_trial, u_test) * ufl.dx
            + dt * Du * ufl.inner(ufl.grad(u_trial), ufl.grad(u_test)) * ufl.dx
        )
        L_u = (
            ufl.inner(self.u_n, u_test) * ufl.dx
            + dt * ufl.inner(reaction_u, u_test) * ufl.dx
        )

        # v-equation: (v_h - v_n)/dt = Dv*laplacian(v_h) + v_n*(1 - b*u_n - v_n - a*w_n)
        reaction_v = self.v_n * (1 - b * self.u_n - self.v_n - a * self.w_n)
        a_v = (
            ufl.inner(v_trial, v_test) * ufl.dx
            + dt * Dv * ufl.inner(ufl.grad(v_trial), ufl.grad(v_test)) * ufl.dx
        )
        L_v = (
            ufl.inner(self.v_n, v_test) * ufl.dx
            + dt * ufl.inner(reaction_v, v_test) * ufl.dx
        )

        # w-equation: (w_h - w_n)/dt = Dw*laplacian(w_h) + w_n*(1 - a*u_n - b*v_n - w_n)
        reaction_w = self.w_n * (1 - a * self.u_n - b * self.v_n - self.w_n)
        a_w = (
            ufl.inner(w_trial, w_test) * ufl.dx
            + dt * Dw * ufl.inner(ufl.grad(w_trial), ufl.grad(w_test)) * ufl.dx
        )
        L_w = (
            ufl.inner(self.w_n, w_test) * ufl.dx
            + dt * ufl.inner(reaction_w, w_test) * ufl.dx
        )

        # Natural boundary condition contributions
        a_u_bc, L_u_bc = build_natural_bc_forms(
            u_trial, u_test, domain_geom, u_boundary_field, params
        )
        if a_u_bc is not None:
            a_u = a_u + dt * a_u_bc
        if L_u_bc is not None:
            L_u = L_u + dt * L_u_bc

        a_v_bc, L_v_bc = build_natural_bc_forms(
            v_trial, v_test, domain_geom, v_boundary_field, params
        )
        if a_v_bc is not None:
            a_v = a_v + dt * a_v_bc
        if L_v_bc is not None:
            L_v = L_v + dt * L_v_bc

        a_w_bc, L_w_bc = build_natural_bc_forms(
            w_trial, w_test, domain_geom, w_boundary_field, params
        )
        if a_w_bc is not None:
            a_w = a_w + dt * a_w_bc
        if L_w_bc is not None:
            L_w = L_w + dt * L_w_bc

        self._dynamic_noise_runtimes = []
        stochastic_u, runtime_u = build_scalar_state_stochastic_term(
            self,
            state_name="u",
            previous_state=self.u_n,
            test=u_test,
            dt=dt,
        )
        if stochastic_u is not None and runtime_u is not None:
            L_u = L_u + stochastic_u
            self._dynamic_noise_runtimes.append(runtime_u)
        stochastic_v, runtime_v = build_scalar_state_stochastic_term(
            self,
            state_name="v",
            previous_state=self.v_n,
            test=v_test,
            dt=dt,
        )
        if stochastic_v is not None and runtime_v is not None:
            L_v = L_v + stochastic_v
            self._dynamic_noise_runtimes.append(runtime_v)
        stochastic_w, runtime_w = build_scalar_state_stochastic_term(
            self,
            state_name="w",
            previous_state=self.w_n,
            test=w_test,
            dt=dt,
        )
        if stochastic_w is not None and runtime_w is not None:
            L_w = L_w + stochastic_w
            self._dynamic_noise_runtimes.append(runtime_w)

        self._u_problem = self.create_linear_problem(
            a_u,
            L_u,
            u=self.u_h,
            bcs=u_bcs,
            petsc_options_prefix="plm_cyclic_competition_u_",
            mpc=u_mpc,
        )
        self._v_problem = self.create_linear_problem(
            a_v,
            L_v,
            u=self.v_h,
            bcs=v_bcs,
            petsc_options_prefix="plm_cyclic_competition_v_",
            mpc=v_mpc,
        )
        self._w_problem = self.create_linear_problem(
            a_w,
            L_w,
            u=self.w_h,
            bcs=w_bcs,
            petsc_options_prefix="plm_cyclic_competition_w_",
            mpc=w_mpc,
        )

    def step(self, t: float, dt: float) -> bool:
        self._u_problem.solve()
        self._v_problem.solve()
        self._w_problem.solve()

        self.u_n.x.array[:] = self.u_h.x.array
        self.u_n.x.scatter_forward()
        self.v_n.x.array[:] = self.v_h.x.array
        self.v_n.x.scatter_forward()
        self.w_n.x.array[:] = self.w_h.x.array
        self.w_n.x.scatter_forward()

        return (
            self._u_problem.solver.getConvergedReason() > 0
            and self._v_problem.solver.getConvergedReason() > 0
            and self._w_problem.solver.getConvergedReason() > 0
        )

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"u": self.u_n, "v": self.v_n, "w": self.w_n}

    def get_num_dofs(self) -> int:
        return self._num_dofs


@register_preset("cyclic_competition")
class CyclicCompetitionPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _CYCLIC_COMPETITION_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _CyclicCompetitionProblem(self.spec, config)

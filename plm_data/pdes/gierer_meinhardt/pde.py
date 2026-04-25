"""Gierer-Meinhardt reaction-diffusion PDE."""

import ufl
from dolfinx import fem

from plm_data.boundary_conditions.runtime import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.initial_conditions.runtime import apply_ic
from plm_data.core.solver_strategies import CONSTANT_LHS_SCALAR_SPD
from plm_data.stochastic import build_scalar_state_stochastic_term
from plm_data.pdes.base import PDE, ProblemInstance, TransientLinearProblem
from plm_data.pdes.boundary_validation import validate_scalar_standard_boundary_field
from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    SATURATING_STOCHASTIC_COUPLINGS,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

_GIERER_MEINHARDT_SPEC = PDESpec(
    name="gierer_meinhardt",
    category="biology",
    description="Gierer-Meinhardt activator-inhibitor reaction-diffusion system.",
    equations={
        "a": "da/dt = Da * laplacian(a) + rho_a * a^2 / h - mu_a * a + sigma_a",
        "h": "tau * dh/dt = Dh * laplacian(h) + rho_h * a^2 - mu_h * h + sigma_h",
    },
    parameters=[
        PDEParameter("Da", "Diffusion coefficient of activator a"),
        PDEParameter("Dh", "Diffusion coefficient of inhibitor h"),
        PDEParameter("rho_a", "Activator self-enhancement rate"),
        PDEParameter("rho_h", "Cross-activation rate for inhibitor"),
        PDEParameter("mu_a", "Activator decay rate"),
        PDEParameter("mu_h", "Inhibitor decay rate"),
        PDEParameter("sigma_a", "Activator basal production"),
        PDEParameter("sigma_h", "Inhibitor basal production"),
        PDEParameter("tau", "Inhibitor time-scale ratio"),
    ],
    inputs={
        "a": InputSpec(
            name="a",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "h": InputSpec(
            name="h",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "a": BoundaryFieldSpec(
            name="a",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for activator a.",
        ),
        "h": BoundaryFieldSpec(
            name="h",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for inhibitor h.",
        ),
    },
    states={
        "a": StateSpec(
            name="a",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
        "h": StateSpec(
            name="h",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
    },
    outputs={
        "a": OutputSpec(
            name="a",
            shape="scalar",
            output_mode="scalar",
            source_name="a",
        ),
        "h": OutputSpec(
            name="h",
            shape="scalar",
            output_mode="scalar",
            source_name="h",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
)


def _space_num_dofs(V: fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


class _GiererMeinhardtProblem(TransientLinearProblem):
    supported_solver_strategies = (CONSTANT_LHS_SCALAR_SPD,)

    def validate_boundary_conditions(self, domain_geom):
        validate_scalar_standard_boundary_field(
            pde_name=self.spec.name,
            field_name="a",
            boundary_field=self.config.boundary_field("a"),
            domain_geom=domain_geom,
        )
        validate_scalar_standard_boundary_field(
            pde_name=self.spec.name,
            field_name="h",
            boundary_field=self.config.boundary_field("h"),
            domain_geom=domain_geom,
        )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        V = fem.functionspace(self.msh, ("Lagrange", 1))

        a_input = self.config.input("a")
        h_input = self.config.input("h")
        a_boundary_field = self.config.boundary_field("a")
        h_boundary_field = self.config.boundary_field("h")

        assert self.config.time is not None
        dt = self.config.time.dt
        Da = self.config.parameters["Da"]
        Dh = self.config.parameters["Dh"]
        rho_a = self.config.parameters["rho_a"]
        rho_h = self.config.parameters["rho_h"]
        mu_a = self.config.parameters["mu_a"]
        mu_h = self.config.parameters["mu_h"]
        sigma_a = self.config.parameters["sigma_a"]
        sigma_h = self.config.parameters["sigma_h"]
        tau = self.config.parameters["tau"]
        if min(Da, Dh, rho_a, rho_h, mu_a, mu_h, tau) <= 0.0:
            raise ValueError(
                "Gierer-Meinhardt diffusion, reaction, decay, and tau parameters "
                "must be positive."
            )

        a_bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            a_boundary_field,
            self.config.parameters,
        )
        h_bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            h_boundary_field,
            self.config.parameters,
        )
        a_mpc = self.create_periodic_constraint(
            V,
            domain_geom,
            a_boundary_field,
            a_bcs,
        )
        h_mpc = self.create_periodic_constraint(
            V,
            domain_geom,
            h_boundary_field,
            h_bcs,
        )

        self.V_a = V if a_mpc is None else a_mpc.function_space
        self.V_h = V if h_mpc is None else h_mpc.function_space
        self._num_dofs = _space_num_dofs(self.V_a) + _space_num_dofs(self.V_h)

        self.a_n = fem.Function(self.V_a, name="a")
        self.h_n = fem.Function(self.V_h, name="h")
        self.a_next = fem.Function(self.V_a, name="a_next")
        self.h_next = fem.Function(self.V_h, name="h_next")

        assert a_input.initial_condition is not None
        apply_ic(
            self.a_n,
            a_input.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        if a_mpc is not None:
            a_mpc.backsubstitution(self.a_n)
        assert h_input.initial_condition is not None
        apply_ic(
            self.h_n,
            h_input.initial_condition,
            self.config.parameters,
            seed=(self.config.seed + 1) if self.config.seed is not None else None,
        )
        if h_mpc is not None:
            h_mpc.backsubstitution(self.h_n)
        self.a_n.x.scatter_forward()
        self.h_n.x.scatter_forward()

        a_trial = ufl.TrialFunction(V)
        a_test = ufl.TestFunction(V)
        h_trial = ufl.TrialFunction(V)
        h_test = ufl.TestFunction(V)

        reaction_a = rho_a * self.a_n**2 / self.h_n + sigma_a
        reaction_h = rho_h * self.a_n**2 + sigma_h

        lhs_a = (
            ufl.inner(a_trial, a_test) * ufl.dx
            + dt * Da * ufl.inner(ufl.grad(a_trial), ufl.grad(a_test)) * ufl.dx
            + dt * mu_a * ufl.inner(a_trial, a_test) * ufl.dx
        )
        rhs_a = (
            ufl.inner(self.a_n, a_test) * ufl.dx
            + dt * ufl.inner(reaction_a, a_test) * ufl.dx
        )

        lhs_h = (
            tau * ufl.inner(h_trial, h_test) * ufl.dx
            + dt * Dh * ufl.inner(ufl.grad(h_trial), ufl.grad(h_test)) * ufl.dx
            + dt * mu_h * ufl.inner(h_trial, h_test) * ufl.dx
        )
        rhs_h = (
            tau * ufl.inner(self.h_n, h_test) * ufl.dx
            + dt * ufl.inner(reaction_h, h_test) * ufl.dx
        )

        lhs_a_bc, rhs_a_bc = build_natural_bc_forms(
            a_trial,
            a_test,
            domain_geom,
            a_boundary_field,
            self.config.parameters,
        )
        if lhs_a_bc is not None:
            lhs_a = lhs_a + dt * lhs_a_bc
        if rhs_a_bc is not None:
            rhs_a = rhs_a + dt * rhs_a_bc

        lhs_h_bc, rhs_h_bc = build_natural_bc_forms(
            h_trial,
            h_test,
            domain_geom,
            h_boundary_field,
            self.config.parameters,
        )
        if lhs_h_bc is not None:
            lhs_h = lhs_h + dt * lhs_h_bc
        if rhs_h_bc is not None:
            rhs_h = rhs_h + dt * rhs_h_bc

        self._dynamic_noise_runtimes = []
        stochastic_a, runtime_a = build_scalar_state_stochastic_term(
            self,
            state_name="a",
            previous_state=self.a_n,
            test=a_test,
            dt=dt,
        )
        if stochastic_a is not None and runtime_a is not None:
            rhs_a = rhs_a + stochastic_a
            self._dynamic_noise_runtimes.append(runtime_a)
        stochastic_h, runtime_h = build_scalar_state_stochastic_term(
            self,
            state_name="h",
            previous_state=self.h_n,
            test=h_test,
            dt=dt,
        )
        if stochastic_h is not None and runtime_h is not None:
            rhs_h = rhs_h + stochastic_h
            self._dynamic_noise_runtimes.append(runtime_h)

        self._a_problem = self.create_linear_problem(
            lhs_a,
            rhs_a,
            u=self.a_next,
            bcs=a_bcs,
            petsc_options_prefix="plm_gierer_meinhardt_a_",
            mpc=a_mpc,
        )
        self._h_problem = self.create_linear_problem(
            lhs_h,
            rhs_h,
            u=self.h_next,
            bcs=h_bcs,
            petsc_options_prefix="plm_gierer_meinhardt_h_",
            mpc=h_mpc,
        )

    def step(self, t: float, dt: float) -> bool:
        self._a_problem.solve()
        self._h_problem.solve()

        self.a_n.x.array[:] = self.a_next.x.array
        self.a_n.x.scatter_forward()
        self.h_n.x.array[:] = self.h_next.x.array
        self.h_n.x.scatter_forward()

        return (
            self._a_problem.solver.getConvergedReason() > 0
            and self._h_problem.solver.getConvergedReason() > 0
        )

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"a": self.a_n, "h": self.h_n}

    def get_num_dofs(self) -> int:
        return self._num_dofs


class GiererMeinhardtPDE(PDE):
    @property
    def spec(self) -> PDESpec:
        return _GIERER_MEINHARDT_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _GiererMeinhardtProblem(self.spec, config)

"""Cancer immunotherapy reaction-diffusion PDE."""

import ufl
from dolfinx import default_real_type, fem

from plm_data.boundary_conditions.runtime import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.initial_conditions.runtime import apply_ic
from plm_data.core.solver_strategies import CONSTANT_LHS_SCALAR_SPD
from plm_data.core.stochastic import build_scalar_state_stochastic_term
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

_IMMUNOTHERAPY_SPEC = PDESpec(
    name="immunotherapy",
    category="biology",
    description=(
        "Three-species cancer immunotherapy model coupling effector cells, "
        "cancer cells, and IL-2 cytokine."
    ),
    equations={
        "u": (
            "du/dt = Du * laplacian(u) + alpha * v - mu_u * u "
            "+ rho_u * u * w / (1 + w) + sigma_u + Ku * t"
        ),
        "v": "dv/dt = Dv * laplacian(v) + v * (1 - v) - u * v / (gamma_v + v)",
        "w": (
            "dw/dt = Dw * laplacian(w) + rho_w * u * v / (gamma_w + v) "
            "- mu_w * w + sigma_w + Kw * t"
        ),
    },
    parameters=[
        PDEParameter("Du", "Diffusion coefficient of effector cells"),
        PDEParameter("Dv", "Diffusion coefficient of cancer cells"),
        PDEParameter("Dw", "Diffusion coefficient of IL-2 cytokine"),
        PDEParameter("alpha", "Effector recruitment rate per cancer density"),
        PDEParameter("mu_u", "Natural death rate of effector cells"),
        PDEParameter("rho_u", "IL-2-stimulated proliferation rate of effectors"),
        PDEParameter("gamma_v", "Cancer-density half-saturation for immune killing"),
        PDEParameter("rho_w", "IL-2 production rate from effector-cancer interaction"),
        PDEParameter("gamma_w", "Cancer-density half-saturation for IL-2 production"),
        PDEParameter("mu_w", "Natural degradation rate of IL-2"),
        PDEParameter("sigma_u", "Basal effector cell infusion rate"),
        PDEParameter("Ku", "Linear-in-time effector cell treatment rate"),
        PDEParameter("sigma_w", "Basal IL-2 infusion rate"),
        PDEParameter("Kw", "Linear-in-time IL-2 treatment rate"),
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
            description="Boundary conditions for effector cells u.",
        ),
        "v": BoundaryFieldSpec(
            name="v",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for cancer cells v.",
        ),
        "w": BoundaryFieldSpec(
            name="w",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for IL-2 cytokine w.",
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
    supported_dimensions=[2],
)


def _space_num_dofs(V: fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


class _ImmunotherapyProblem(TransientLinearProblem):
    supported_solver_strategies = (CONSTANT_LHS_SCALAR_SPD,)

    def validate_boundary_conditions(self, domain_geom):
        for field_name in ("u", "v", "w"):
            validate_scalar_standard_boundary_field(
                pde_name=self.spec.name,
                field_name=field_name,
                boundary_field=self.config.boundary_field(field_name),
                domain_geom=domain_geom,
            )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        V = fem.functionspace(self.msh, ("Lagrange", 1))

        assert self.config.time is not None
        dt = self.config.time.dt
        params = self.config.parameters
        Du = params["Du"]
        Dv = params["Dv"]
        Dw = params["Dw"]
        alpha = params["alpha"]
        mu_u = params["mu_u"]
        rho_u = params["rho_u"]
        gamma_v = params["gamma_v"]
        rho_w = params["rho_w"]
        gamma_w = params["gamma_w"]
        mu_w = params["mu_w"]
        sigma_u = params["sigma_u"]
        Ku = params["Ku"]
        sigma_w = params["sigma_w"]
        Kw = params["Kw"]
        if min(Du, Dv, Dw, alpha, mu_u, rho_u, gamma_v, rho_w, gamma_w, mu_w) <= 0.0:
            raise ValueError(
                "Immunotherapy diffusion, reaction, decay, and saturation "
                "parameters must be positive."
            )
        if min(sigma_u, Ku, sigma_w, Kw) < 0.0:
            raise ValueError("Immunotherapy treatment parameters cannot be negative.")

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

        self._t_const = fem.Constant(self.msh, default_real_type(0.0))

        u_trial = ufl.TrialFunction(V)
        u_test = ufl.TestFunction(V)
        v_trial = ufl.TrialFunction(V)
        v_test = ufl.TestFunction(V)
        w_trial = ufl.TrialFunction(V)
        w_test = ufl.TestFunction(V)

        lhs_u = (
            ufl.inner(u_trial, u_test) * ufl.dx
            + dt * Du * ufl.inner(ufl.grad(u_trial), ufl.grad(u_test)) * ufl.dx
            + dt * mu_u * ufl.inner(u_trial, u_test) * ufl.dx
        )
        reaction_u = (
            alpha * self.v_n
            + rho_u * self.u_n * self.w_n / (1 + self.w_n)
            + sigma_u
            + Ku * self._t_const
        )
        rhs_u = (
            ufl.inner(self.u_n, u_test) * ufl.dx
            + dt * ufl.inner(reaction_u, u_test) * ufl.dx
        )

        lhs_v = (
            ufl.inner(v_trial, v_test) * ufl.dx
            + dt * Dv * ufl.inner(ufl.grad(v_trial), ufl.grad(v_test)) * ufl.dx
        )
        reaction_v = self.v_n * (1 - self.v_n) - self.u_n * self.v_n / (
            gamma_v + self.v_n
        )
        rhs_v = (
            ufl.inner(self.v_n, v_test) * ufl.dx
            + dt * ufl.inner(reaction_v, v_test) * ufl.dx
        )

        lhs_w = (
            ufl.inner(w_trial, w_test) * ufl.dx
            + dt * Dw * ufl.inner(ufl.grad(w_trial), ufl.grad(w_test)) * ufl.dx
            + dt * mu_w * ufl.inner(w_trial, w_test) * ufl.dx
        )
        reaction_w = (
            rho_w * self.u_n * self.v_n / (gamma_w + self.v_n)
            + sigma_w
            + Kw * self._t_const
        )
        rhs_w = (
            ufl.inner(self.w_n, w_test) * ufl.dx
            + dt * ufl.inner(reaction_w, w_test) * ufl.dx
        )

        lhs_u_bc, rhs_u_bc = build_natural_bc_forms(
            u_trial,
            u_test,
            domain_geom,
            u_boundary_field,
            params,
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
            params,
        )
        if lhs_v_bc is not None:
            lhs_v = lhs_v + dt * lhs_v_bc
        if rhs_v_bc is not None:
            rhs_v = rhs_v + dt * rhs_v_bc

        lhs_w_bc, rhs_w_bc = build_natural_bc_forms(
            w_trial,
            w_test,
            domain_geom,
            w_boundary_field,
            params,
        )
        if lhs_w_bc is not None:
            lhs_w = lhs_w + dt * lhs_w_bc
        if rhs_w_bc is not None:
            rhs_w = rhs_w + dt * rhs_w_bc

        self._dynamic_noise_runtimes = []
        stochastic_u, runtime_u = build_scalar_state_stochastic_term(
            self,
            state_name="u",
            previous_state=self.u_n,
            test=u_test,
            dt=dt,
        )
        if stochastic_u is not None and runtime_u is not None:
            rhs_u = rhs_u + stochastic_u
            self._dynamic_noise_runtimes.append(runtime_u)
        stochastic_v, runtime_v = build_scalar_state_stochastic_term(
            self,
            state_name="v",
            previous_state=self.v_n,
            test=v_test,
            dt=dt,
        )
        if stochastic_v is not None and runtime_v is not None:
            rhs_v = rhs_v + stochastic_v
            self._dynamic_noise_runtimes.append(runtime_v)
        stochastic_w, runtime_w = build_scalar_state_stochastic_term(
            self,
            state_name="w",
            previous_state=self.w_n,
            test=w_test,
            dt=dt,
        )
        if stochastic_w is not None and runtime_w is not None:
            rhs_w = rhs_w + stochastic_w
            self._dynamic_noise_runtimes.append(runtime_w)

        self._u_problem = self.create_linear_problem(
            lhs_u,
            rhs_u,
            u=self.u_h,
            bcs=u_bcs,
            petsc_options_prefix="plm_immunotherapy_u_",
            mpc=u_mpc,
        )
        self._v_problem = self.create_linear_problem(
            lhs_v,
            rhs_v,
            u=self.v_h,
            bcs=v_bcs,
            petsc_options_prefix="plm_immunotherapy_v_",
            mpc=v_mpc,
        )
        self._w_problem = self.create_linear_problem(
            lhs_w,
            rhs_w,
            u=self.w_h,
            bcs=w_bcs,
            petsc_options_prefix="plm_immunotherapy_w_",
            mpc=w_mpc,
        )

    def step(self, t: float, dt: float) -> bool:
        self._t_const.value = default_real_type(t + dt)

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


class ImmunotherapyPDE(PDE):
    @property
    def spec(self) -> PDESpec:
        return _IMMUNOTHERAPY_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _ImmunotherapyProblem(self.spec, config)

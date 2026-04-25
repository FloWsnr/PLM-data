"""Gray-Scott reaction-diffusion PDE."""

import ufl
from dolfinx import default_real_type, fem

from plm_data.boundary_conditions.runtime import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.initial_conditions.runtime import apply_ic
from plm_data.core.solver_strategies import (
    CONSTANT_LHS_SCALAR_NONSYMMETRIC,
    CONSTANT_LHS_SCALAR_SPD,
)
from plm_data.core.spatial_fields import (
    build_vector_ufl_field,
    is_exact_zero_field_expression,
)
from plm_data.core.stochastic import build_scalar_state_stochastic_term
from plm_data.pdes.base import PDE, ProblemInstance, TransientLinearProblem
from plm_data.pdes.boundary_validation import validate_scalar_standard_boundary_field
from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    CoefficientSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    SATURATING_STOCHASTIC_COUPLINGS,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

_GRAY_SCOTT_SPEC = PDESpec(
    name="gray_scott",
    category="physics",
    description="Gray-Scott reaction-diffusion system for substrate/autocatalyst patterns.",
    equations={
        "u": "du/dt + velocity dot grad(u) = Du * laplacian(u) - u * v^2 + F * (1 - u)",
        "v": "dv/dt = Dv * laplacian(v) + u * v^2 - (F + k) * v",
    },
    parameters=[
        PDEParameter("Du", "Diffusion coefficient of substrate u"),
        PDEParameter("Dv", "Diffusion coefficient of autocatalyst v"),
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
            description="Boundary conditions for substrate u.",
        ),
        "v": BoundaryFieldSpec(
            name="v",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for autocatalyst v.",
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
    supported_dimensions=[2],
    coefficients={
        "velocity": CoefficientSpec(
            name="velocity",
            shape="vector",
            description="Prescribed advection velocity applied to substrate u.",
        )
    },
)


def _space_num_dofs(V: fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


class _GrayScottProblem(TransientLinearProblem):
    supported_solver_strategies = (
        CONSTANT_LHS_SCALAR_SPD,
        CONSTANT_LHS_SCALAR_NONSYMMETRIC,
    )

    def _validate_solver_strategy(self) -> None:
        super()._validate_solver_strategy()
        has_advection = not is_exact_zero_field_expression(
            self.config.coefficient("velocity"),
            self.config.parameters,
        )
        if (
            has_advection
            and self.config.solver.strategy != CONSTANT_LHS_SCALAR_NONSYMMETRIC
        ):
            raise ValueError(
                "Gray-Scott with nonzero velocity requires solver strategy "
                f"'{CONSTANT_LHS_SCALAR_NONSYMMETRIC}'."
            )

    def validate_boundary_conditions(self, domain_geom):
        validate_scalar_standard_boundary_field(
            pde_name=self.spec.name,
            field_name="u",
            boundary_field=self.config.boundary_field("u"),
            domain_geom=domain_geom,
        )
        validate_scalar_standard_boundary_field(
            pde_name=self.spec.name,
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

        assert self.config.time is not None
        dt = self.config.time.dt
        Du = self.config.parameters["Du"]
        Dv = self.config.parameters["Dv"]
        F = self.config.parameters["F"]
        k = self.config.parameters["k"]
        if Du <= 0.0 or Dv <= 0.0:
            raise ValueError("Gray-Scott diffusion parameters must be positive.")
        if F <= 0.0 or k <= 0.0:
            raise ValueError("Gray-Scott feed and kill rates must be positive.")

        dt_c = fem.Constant(self.msh, default_real_type(dt))
        velocity = build_vector_ufl_field(
            self.msh,
            self.config.coefficient("velocity"),
            self.config.parameters,
        )
        if velocity is None:
            raise ValueError(
                "Gray-Scott coefficient 'velocity' cannot use a custom expression"
            )
        has_advection = not is_exact_zero_field_expression(
            self.config.coefficient("velocity"),
            self.config.parameters,
        )

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
        assert v_input.initial_condition is not None
        apply_ic(
            self.v_n,
            v_input.initial_condition,
            self.config.parameters,
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
            + dt_c * Du * ufl.inner(ufl.grad(u_trial), ufl.grad(u_test)) * ufl.dx
            + dt_c * F * ufl.inner(u_trial, u_test) * ufl.dx
        )
        L_u = (
            ufl.inner(self.u_n, u_test) * ufl.dx
            + dt_c * ufl.inner(F - cubic_reaction, u_test) * ufl.dx
        )

        if has_advection:
            a_u = a_u + dt_c * ufl.inner(velocity, ufl.grad(u_trial)) * u_test * ufl.dx

        a_v = (
            ufl.inner(v_trial, v_test) * ufl.dx
            + dt_c * Dv * ufl.inner(ufl.grad(v_trial), ufl.grad(v_test)) * ufl.dx
            + dt_c * (F + k) * ufl.inner(v_trial, v_test) * ufl.dx
        )
        L_v = (
            ufl.inner(self.v_n, v_test) * ufl.dx
            + dt_c * ufl.inner(cubic_reaction, v_test) * ufl.dx
        )

        self._dynamic_noise_runtimes = []
        u_stochastic_term, u_stochastic_runtime = build_scalar_state_stochastic_term(
            self,
            state_name="u",
            previous_state=self.u_n,
            test=u_test,
            dt=dt,
        )
        if u_stochastic_term is not None and u_stochastic_runtime is not None:
            L_u = L_u + u_stochastic_term
            self._dynamic_noise_runtimes.append(u_stochastic_runtime)

        v_stochastic_term, v_stochastic_runtime = build_scalar_state_stochastic_term(
            self,
            state_name="v",
            previous_state=self.v_n,
            test=v_test,
            dt=dt,
        )
        if v_stochastic_term is not None and v_stochastic_runtime is not None:
            L_v = L_v + v_stochastic_term
            self._dynamic_noise_runtimes.append(v_stochastic_runtime)

        if has_advection:
            h = ufl.CellDiameter(self.msh)
            velocity_norm = ufl.sqrt(ufl.inner(velocity, velocity))
            zero = fem.Constant(self.msh, default_real_type(0.0))
            stabilized_diffusivity = ufl.max_value(ufl.as_ufl(Du), zero)
            tau = 1.0 / ufl.sqrt(
                (2.0 / dt_c) ** 2
                + (2.0 * velocity_norm / h) ** 2
                + (4.0 * stabilized_diffusivity / (h**2)) ** 2
            )
            streamline_test = ufl.inner(velocity, ufl.grad(u_test))
            lhs_residual = (
                u_trial / dt_c
                + ufl.inner(velocity, ufl.grad(u_trial))
                - ufl.div(Du * ufl.grad(u_trial))
                + F * u_trial
            )
            rhs_residual = self.u_n / dt_c + F - cubic_reaction
            a_u = a_u + dt_c * tau * lhs_residual * streamline_test * ufl.dx
            L_u = L_u + dt_c * tau * rhs_residual * streamline_test * ufl.dx

        a_u_bc, L_u_bc = build_natural_bc_forms(
            u_trial,
            u_test,
            domain_geom,
            u_boundary_field,
            self.config.parameters,
        )
        if a_u_bc is not None:
            a_u = a_u + dt_c * a_u_bc
        if L_u_bc is not None:
            L_u = L_u + dt_c * L_u_bc

        a_v_bc, L_v_bc = build_natural_bc_forms(
            v_trial,
            v_test,
            domain_geom,
            v_boundary_field,
            self.config.parameters,
        )
        if a_v_bc is not None:
            a_v = a_v + dt_c * a_v_bc
        if L_v_bc is not None:
            L_v = L_v + dt_c * L_v_bc

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


class GrayScottPDE(PDE):
    @property
    def spec(self) -> PDESpec:
        return _GRAY_SCOTT_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _GrayScottProblem(self.spec, config)

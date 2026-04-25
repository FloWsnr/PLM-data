"""Shared scalar reaction-diffusion helper for biology PDEs."""

from abc import ABC, abstractmethod

import ufl
from dolfinx import default_real_type, fem

from plm_data.boundary_conditions.runtime import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.initial_conditions.runtime import apply_ic
from plm_data.core.solver_strategies import (
    NONLINEAR_MIXED_DIRECT,
)
from plm_data.core.stochastic import build_scalar_state_stochastic_term
from plm_data.core.spatial_fields import (
    build_vector_ufl_field,
    is_exact_zero_field_expression,
)
from plm_data.pdes.base import TransientNonlinearProblem
from plm_data.pdes.boundary_validation import (
    validate_scalar_standard_boundary_field,
)
from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    CoefficientSpec,
    GENERIC_STOCHASTIC_COUPLINGS,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)


def build_scalar_reaction_diffusion_spec(
    *,
    name: str,
    description: str,
    reaction_equation: str,
    parameters: list[PDEParameter],
) -> PDESpec:
    """Build a standard scalar reaction-diffusion PDE spec."""
    return PDESpec(
        name=name,
        category="biology",
        description=description,
        equations={
            "u": (f"du/dt + velocity·grad(u) = D * laplacian(u) + {reaction_equation}")
        },
        parameters=[PDEParameter("D", "Diffusion coefficient"), *parameters],
        inputs={
            "u": InputSpec(
                name="u",
                shape="scalar",
                allow_source=False,
                allow_initial_condition=True,
            )
        },
        boundary_fields={
            "u": BoundaryFieldSpec(
                name="u",
                shape="scalar",
                operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
                description="Boundary conditions for the scalar population field.",
            )
        },
        states={
            "u": StateSpec(
                name="u",
                shape="scalar",
                stochastic_couplings=GENERIC_STOCHASTIC_COUPLINGS,
            )
        },
        outputs={
            "u": OutputSpec(
                name="u",
                shape="scalar",
                output_mode="scalar",
                source_name="u",
            )
        },
        static_fields=[],
        supported_dimensions=[2],
        coefficients={
            "velocity": CoefficientSpec(
                name="velocity",
                shape="vector",
                description="Prescribed advection velocity field.",
            )
        },
    )


class ScalarReactionDiffusionProblem(TransientNonlinearProblem, ABC):
    """Shared nonlinear scalar reaction-diffusion time stepper."""

    supported_solver_strategies = (NONLINEAR_MIXED_DIRECT,)

    def validate_boundary_conditions(self, domain_geom):
        validate_scalar_standard_boundary_field(
            pde_name=self.spec.name,
            field_name="u",
            boundary_field=self.config.boundary_field("u"),
            domain_geom=domain_geom,
        )

    @abstractmethod
    def reaction_term(self, u_current):
        """Return the reaction term evaluated on the current state."""

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        V = fem.functionspace(self.msh, ("Lagrange", 1))

        u_input = self.config.input("u")
        boundary_field = self.config.boundary_field("u")
        dt = self.config.time.dt
        diffusion = self.config.parameters["D"]

        self._has_advection = not is_exact_zero_field_expression(
            self.config.coefficient("velocity"),
            self.config.parameters,
        )
        velocity = build_vector_ufl_field(
            self.msh,
            self.config.coefficient("velocity"),
            self.config.parameters,
        )
        if velocity is None:
            raise ValueError(
                f"PDE '{self.spec.name}' coefficient 'velocity' cannot use a "
                "custom expression"
            )

        bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            boundary_field,
            self.config.parameters,
        )
        mpc = self.create_periodic_constraint(V, domain_geom, boundary_field, bcs)
        self.V = V if mpc is None else mpc.function_space
        self._num_dofs = (
            self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs
        )

        self.u = fem.Function(self.V, name="u")
        self.u_n = fem.Function(self.V, name="u_prev")

        initial_condition = u_input.initial_condition
        assert initial_condition is not None
        apply_ic(
            self.u,
            initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        self.u.x.scatter_forward()
        self.u_n.x.array[:] = self.u.x.array
        self.u_n.x.scatter_forward()

        w = ufl.TestFunction(V)
        dt_c = fem.Constant(self.msh, default_real_type(dt))
        diffusion_c = fem.Constant(self.msh, default_real_type(diffusion))

        F = (
            ufl.inner((self.u - self.u_n) / dt_c, w) * ufl.dx
            + diffusion_c * ufl.inner(ufl.grad(self.u), ufl.grad(w)) * ufl.dx
            - ufl.inner(self.reaction_term(self.u), w) * ufl.dx
        )
        if self._has_advection:
            F = F + ufl.inner(velocity, ufl.grad(self.u)) * w * ufl.dx

        a_bc, L_bc = build_natural_bc_forms(
            self.u,
            w,
            domain_geom,
            boundary_field,
            self.config.parameters,
        )
        if a_bc is not None:
            F = F + a_bc
        if L_bc is not None:
            F = F - L_bc

        if self._has_advection:
            h = ufl.CellDiameter(self.msh)
            zero = fem.Constant(self.msh, default_real_type(0.0))
            velocity_norm = ufl.sqrt(ufl.inner(velocity, velocity))
            stabilized_diffusivity = ufl.max_value(diffusion_c, zero)
            tau = 1.0 / ufl.sqrt(
                (2.0 / dt_c) ** 2
                + (2.0 * velocity_norm / h) ** 2
                + (4.0 * stabilized_diffusivity / (h**2)) ** 2
            )
            streamline_test = ufl.inner(velocity, ufl.grad(w))
            strong_residual = (
                (self.u - self.u_n) / dt_c
                + ufl.inner(velocity, ufl.grad(self.u))
                - ufl.div(diffusion_c * ufl.grad(self.u))
                - self.reaction_term(self.u)
            )
            F = F + dt_c * tau * strong_residual * streamline_test * ufl.dx

        stochastic_term, stochastic_runtime = build_scalar_state_stochastic_term(
            self,
            state_name="u",
            previous_state=self.u_n,
            test=w,
            dt=dt,
        )
        self._dynamic_noise_runtimes = []
        if stochastic_term is not None and stochastic_runtime is not None:
            F = F - stochastic_term
            self._dynamic_noise_runtimes.append(stochastic_runtime)

        J = ufl.derivative(F, self.u, ufl.TrialFunction(V))

        self.problem = self.create_nonlinear_problem(
            F,
            self.u,
            bcs=bcs,
            petsc_options_prefix=f"plm_{self.spec.name}_",
            J=J,
            mpc=mpc,
        )

    def step(self, t: float, dt: float) -> bool:
        self.u_n.x.array[:] = self.u.x.array
        self.u_n.x.scatter_forward()
        self.problem.solve()
        self.u.x.scatter_forward()
        return self.problem.solver.getConvergedReason() > 0

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"u": self.u}

    def get_num_dofs(self) -> int:
        return self._num_dofs

"""Transient Darcy flow in porous media with passive tracer transport."""

import ufl
from dolfinx import default_real_type, fem

from plm_data.boundary_conditions.runtime import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.initial_conditions.runtime import apply_ic
from plm_data.core.solver_strategies import (
    CONSTANT_LHS_SCALAR_NONSYMMETRIC,
    TRANSIENT_MIXED_DIRECT,
)
from plm_data.stochastic import (
    build_scalar_coefficient,
    build_scalar_state_stochastic_term,
)
from plm_data.fields import (
    build_ufl_field,
    is_exact_zero_field_expression,
    scalar_expression_to_config,
)
from plm_data.pdes.base import PDE, ProblemInstance, TransientLinearProblem
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

_DARCY_SPEC = PDESpec(
    name="darcy",
    category="fluids",
    description=(
        "Transient Darcy pressure diffusion in porous media with a passive "
        "tracer transported by the Darcy velocity."
    ),
    equations={
        "pressure": "storage * ∂pressure/∂t = ∇·(mobility ∇pressure) + q_p",
        "velocity": "velocity = -mobility ∇pressure",
        "concentration": (
            "porosity * ∂concentration/∂t + velocity·∇concentration = "
            "∇·(dispersion ∇concentration) + q_c"
        ),
    },
    parameters=[
        PDEParameter("storage", "Storage coefficient controlling pressure relaxation"),
        PDEParameter("porosity", "Pore volume fraction for tracer storage"),
    ],
    inputs={
        "pressure": InputSpec(
            name="pressure",
            shape="scalar",
            allow_source=True,
            allow_initial_condition=True,
        ),
        "concentration": InputSpec(
            name="concentration",
            shape="scalar",
            allow_source=True,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "pressure": BoundaryFieldSpec(
            name="pressure",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the pressure field.",
        ),
        "concentration": BoundaryFieldSpec(
            name="concentration",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the passive tracer field.",
        ),
    },
    states={
        "pressure": StateSpec(
            name="pressure",
            shape="scalar",
            stochastic_couplings=GENERIC_STOCHASTIC_COUPLINGS,
        ),
        "concentration": StateSpec(
            name="concentration",
            shape="scalar",
            stochastic_couplings=GENERIC_STOCHASTIC_COUPLINGS,
        ),
    },
    outputs={
        "pressure": OutputSpec(
            name="pressure",
            shape="scalar",
            output_mode="scalar",
            source_name="pressure",
        ),
        "concentration": OutputSpec(
            name="concentration",
            shape="scalar",
            output_mode="scalar",
            source_name="concentration",
        ),
        "velocity": OutputSpec(
            name="velocity",
            shape="vector",
            output_mode="components",
            source_name="velocity",
            source_kind="derived",
        ),
        "speed": OutputSpec(
            name="speed",
            shape="scalar",
            output_mode="scalar",
            source_name="speed",
            source_kind="derived",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
    coefficients={
        "mobility": CoefficientSpec(
            name="mobility",
            shape="scalar",
            description="Darcy mobility field, e.g. permeability divided by viscosity.",
            allow_randomization=True,
        ),
        "dispersion": CoefficientSpec(
            name="dispersion",
            shape="scalar",
            description="Tracer diffusion/dispersion coefficient field.",
            allow_randomization=True,
        ),
    },
)


def _space_num_dofs(V: fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


class _DarcyProblem(TransientLinearProblem):
    supported_solver_strategies = (
        CONSTANT_LHS_SCALAR_NONSYMMETRIC,
        TRANSIENT_MIXED_DIRECT,
    )

    def validate_boundary_conditions(self, domain_geom):
        pressure_boundary_field = self.config.boundary_field("pressure")
        concentration_boundary_field = self.config.boundary_field("concentration")

        validate_scalar_standard_boundary_field(
            pde_name=self.spec.name,
            field_name="pressure",
            boundary_field=pressure_boundary_field,
            domain_geom=domain_geom,
        )
        validate_scalar_standard_boundary_field(
            pde_name=self.spec.name,
            field_name="concentration",
            boundary_field=concentration_boundary_field,
            domain_geom=domain_geom,
        )

        if (
            pressure_boundary_field.periodic_pair_keys()
            != concentration_boundary_field.periodic_pair_keys()
        ):
            raise ValueError(
                "Pressure and concentration boundary conditions must use identical "
                "periodic side pairs."
            )

    def should_reuse_linear_lhs(self, *, mpc=None) -> bool:
        # The tracer operator depends on the current Darcy velocity, so its matrix
        # changes whenever the pressure field changes.
        return False

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh

        storage = self.config.parameters["storage"]
        porosity = self.config.parameters["porosity"]
        if storage <= 0.0:
            raise ValueError("Darcy parameter 'storage' must be positive.")
        if porosity <= 0.0 or porosity > 1.0:
            raise ValueError("Darcy parameter 'porosity' must lie in (0, 1].")

        pressure_input = self.config.input("pressure")
        concentration_input = self.config.input("concentration")
        pressure_boundary_field = self.config.boundary_field("pressure")
        concentration_boundary_field = self.config.boundary_field("concentration")

        V = fem.functionspace(self.msh, ("Lagrange", 1))

        pressure_bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            pressure_boundary_field,
            self.config.parameters,
        )
        concentration_bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            concentration_boundary_field,
            self.config.parameters,
        )

        pressure_mpc = self.create_periodic_constraint(
            V,
            domain_geom,
            pressure_boundary_field,
            pressure_bcs,
        )
        concentration_mpc = self.create_periodic_constraint(
            V,
            domain_geom,
            concentration_boundary_field,
            concentration_bcs,
        )

        self.V_pressure = V if pressure_mpc is None else pressure_mpc.function_space
        self.V_concentration = (
            V if concentration_mpc is None else concentration_mpc.function_space
        )
        self._num_dofs = _space_num_dofs(self.V_pressure) + _space_num_dofs(
            self.V_concentration
        )

        self.pressure_n = fem.Function(self.V_pressure, name="pressure")
        self.pressure_h = fem.Function(self.V_pressure, name="pressure_next")
        self.concentration_n = fem.Function(
            self.V_concentration,
            name="concentration",
        )
        self.concentration_h = fem.Function(
            self.V_concentration,
            name="concentration_next",
        )

        assert pressure_input.initial_condition is not None
        apply_ic(
            self.pressure_n,
            pressure_input.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        self.pressure_n.x.scatter_forward()

        assert concentration_input.initial_condition is not None
        concentration_seed = (
            self.config.seed + 1 if self.config.seed is not None else None
        )
        apply_ic(
            self.concentration_n,
            concentration_input.initial_condition,
            self.config.parameters,
            seed=concentration_seed,
        )
        self.concentration_n.x.scatter_forward()

        dt_c = fem.Constant(self.msh, default_real_type(self.config.time.dt))
        storage_c = fem.Constant(self.msh, default_real_type(storage))
        porosity_c = fem.Constant(self.msh, default_real_type(porosity))

        mobility = build_scalar_coefficient(self, "mobility")
        if mobility is None:
            raise ValueError(
                "Darcy coefficient 'mobility' cannot use a custom expression"
            )

        dispersion = build_scalar_coefficient(self, "dispersion")
        if dispersion is None:
            raise ValueError(
                "Darcy coefficient 'dispersion' cannot use a custom expression"
            )

        assert pressure_input.source is not None
        pressure_source_is_zero = is_exact_zero_field_expression(
            pressure_input.source,
            self.config.parameters,
        )
        pressure_source = build_ufl_field(
            self.msh,
            scalar_expression_to_config(pressure_input.source),
            self.config.parameters,
        )
        if pressure_source is None:
            raise ValueError("Darcy pressure source cannot use a custom expression")

        assert concentration_input.source is not None
        concentration_source_is_zero = is_exact_zero_field_expression(
            concentration_input.source,
            self.config.parameters,
        )
        concentration_source = build_ufl_field(
            self.msh,
            scalar_expression_to_config(concentration_input.source),
            self.config.parameters,
        )
        if concentration_source is None:
            raise ValueError(
                "Darcy concentration source cannot use a custom expression"
            )

        pressure_trial = ufl.TrialFunction(V)
        pressure_test = ufl.TestFunction(V)
        a_pressure = (
            storage_c * ufl.inner(pressure_trial, pressure_test) * ufl.dx
            + dt_c
            * ufl.inner(mobility * ufl.grad(pressure_trial), ufl.grad(pressure_test))
            * ufl.dx
        )
        L_pressure = storage_c * ufl.inner(self.pressure_n, pressure_test) * ufl.dx
        if not pressure_source_is_zero:
            L_pressure = L_pressure + dt_c * pressure_source * pressure_test * ufl.dx

        pressure_stochastic_term, pressure_stochastic_runtime = (
            build_scalar_state_stochastic_term(
                self,
                state_name="pressure",
                previous_state=self.pressure_n,
                test=pressure_test,
                dt=self.config.time.dt,
            )
        )
        self._dynamic_noise_runtimes = []
        if (
            pressure_stochastic_term is not None
            and pressure_stochastic_runtime is not None
        ):
            L_pressure = L_pressure + pressure_stochastic_term
            self._dynamic_noise_runtimes.append(pressure_stochastic_runtime)

        a_pressure_bc, L_pressure_bc = build_natural_bc_forms(
            pressure_trial,
            pressure_test,
            domain_geom,
            pressure_boundary_field,
            self.config.parameters,
        )
        if a_pressure_bc is not None:
            a_pressure = a_pressure + dt_c * a_pressure_bc
        if L_pressure_bc is not None:
            L_pressure = L_pressure + dt_c * L_pressure_bc

        self._pressure_problem = self.create_linear_problem(
            a_pressure,
            L_pressure,
            u=self.pressure_h,
            bcs=pressure_bcs,
            petsc_options_prefix="plm_darcy_pressure_",
            mpc=pressure_mpc,
        )

        self._velocity_expr = -mobility * ufl.grad(self.pressure_n)

        concentration_trial = ufl.TrialFunction(V)
        concentration_test = ufl.TestFunction(V)
        advection_term = ufl.inner(self._velocity_expr, ufl.grad(concentration_trial))
        diffusion_term = ufl.inner(
            dispersion * ufl.grad(concentration_trial),
            ufl.grad(concentration_test),
        )
        a_concentration = (
            porosity_c * ufl.inner(concentration_trial, concentration_test) * ufl.dx
            + dt_c * (advection_term * concentration_test + diffusion_term) * ufl.dx
        )
        L_concentration = (
            porosity_c * ufl.inner(self.concentration_n, concentration_test) * ufl.dx
        )
        if not concentration_source_is_zero:
            L_concentration = (
                L_concentration
                + dt_c * concentration_source * concentration_test * ufl.dx
            )

        concentration_stochastic_term, concentration_stochastic_runtime = (
            build_scalar_state_stochastic_term(
                self,
                state_name="concentration",
                previous_state=self.concentration_n,
                test=concentration_test,
                dt=self.config.time.dt,
            )
        )
        if (
            concentration_stochastic_term is not None
            and concentration_stochastic_runtime is not None
        ):
            L_concentration = L_concentration + concentration_stochastic_term
            self._dynamic_noise_runtimes.append(concentration_stochastic_runtime)

        a_concentration_bc, L_concentration_bc = build_natural_bc_forms(
            concentration_trial,
            concentration_test,
            domain_geom,
            concentration_boundary_field,
            self.config.parameters,
        )
        if a_concentration_bc is not None:
            a_concentration = a_concentration + dt_c * a_concentration_bc
        if L_concentration_bc is not None:
            L_concentration = L_concentration + dt_c * L_concentration_bc

        h = ufl.CellDiameter(self.msh)
        zero = fem.Constant(self.msh, default_real_type(0.0))
        speed = ufl.sqrt(ufl.inner(self._velocity_expr, self._velocity_expr))
        stabilized_dispersion = ufl.max_value(dispersion, zero)
        tau = 1.0 / ufl.sqrt(
            (2.0 * porosity_c / dt_c) ** 2
            + (2.0 * speed / h) ** 2
            + (4.0 * stabilized_dispersion / (h**2)) ** 2
        )
        streamline_test = ufl.inner(self._velocity_expr, ufl.grad(concentration_test))
        lhs_residual = (
            porosity_c * concentration_trial / dt_c
            + ufl.inner(self._velocity_expr, ufl.grad(concentration_trial))
            - ufl.div(dispersion * ufl.grad(concentration_trial))
        )
        rhs_residual = porosity_c * self.concentration_n / dt_c + concentration_source
        a_concentration = (
            a_concentration + dt_c * tau * lhs_residual * streamline_test * ufl.dx
        )
        L_concentration = (
            L_concentration + dt_c * tau * rhs_residual * streamline_test * ufl.dx
        )

        self._concentration_problem = self.create_linear_problem(
            a_concentration,
            L_concentration,
            u=self.concentration_h,
            bcs=concentration_bcs,
            petsc_options_prefix="plm_darcy_concentration_",
            mpc=concentration_mpc,
        )

        gdim = self.msh.geometry.dim
        velocity_output_space = fem.functionspace(
            self.msh,
            ("Discontinuous Lagrange", 1, (gdim,)),
        )
        scalar_output_space = fem.functionspace(self.msh, ("Discontinuous Lagrange", 1))
        self.velocity_out = fem.Function(velocity_output_space, name="velocity")
        self.speed_out = fem.Function(scalar_output_space, name="speed")
        self._velocity_output_expr = fem.Expression(
            self._velocity_expr,
            velocity_output_space.element.interpolation_points,
        )
        self._speed_output_expr = fem.Expression(
            speed,
            scalar_output_space.element.interpolation_points,
        )

    def _update_derived_outputs(self) -> None:
        self.velocity_out.interpolate(self._velocity_output_expr)
        self.speed_out.interpolate(self._speed_output_expr)

    def step(self, t: float, dt: float) -> bool:
        self._pressure_problem.solve()
        pressure_converged = self._pressure_problem.solver.getConvergedReason() > 0
        if not pressure_converged:
            return False

        self.pressure_n.x.array[:] = self.pressure_h.x.array
        self.pressure_n.x.scatter_forward()

        self._concentration_problem.solve()
        concentration_converged = (
            self._concentration_problem.solver.getConvergedReason() > 0
        )
        if not concentration_converged:
            return False

        self.concentration_n.x.array[:] = self.concentration_h.x.array
        self.concentration_n.x.scatter_forward()
        return True

    def get_output_fields(self) -> dict[str, fem.Function]:
        self._update_derived_outputs()
        return {
            "pressure": self.pressure_n,
            "concentration": self.concentration_n,
            "velocity": self.velocity_out,
            "speed": self.speed_out,
        }

    def get_num_dofs(self) -> int:
        return self._num_dofs


class DarcyPDE(PDE):
    @property
    def spec(self) -> PDESpec:
        return _DARCY_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _DarcyProblem(self.spec, config)

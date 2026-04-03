"""Smooth compressible Navier-Stokes preset using primitive variables."""

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem
from mpi4py import MPI

from plm_data.core.boundary_conditions import (
    apply_dirichlet_bcs_to_subspace,
    apply_vector_dirichlet_bcs_to_subspace,
    build_natural_bc_forms,
    build_vector_natural_bc_forms,
)
from plm_data.core.initial_conditions import apply_ic, apply_vector_ic
from plm_data.core.solver_strategies import NONLINEAR_MIXED_DIRECT
from plm_data.core.spatial_fields import (
    build_ufl_field,
    build_vector_ufl_field,
    scalar_expression_to_config,
)
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, TransientNonlinearProblem
from plm_data.presets.boundary_validation import (
    validate_boundary_field_structure,
    validate_scalar_standard_boundary_field,
    validate_vector_standard_boundary_field,
)
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
    VECTOR_STANDARD_BOUNDARY_OPERATORS,
)

_DENSITY_BOUNDARY_OPERATORS = {
    "dirichlet": SCALAR_STANDARD_BOUNDARY_OPERATORS["dirichlet"],
    "periodic": SCALAR_STANDARD_BOUNDARY_OPERATORS["periodic"],
}

_COMPRESSIBLE_NAVIER_STOKES_SPEC = PresetSpec(
    name="compressible_navier_stokes",
    category="fluids",
    description=(
        "Smooth ideal-gas compressible Navier-Stokes flow in primitive variables "
        "(density, velocity, temperature) with heat conduction."
    ),
    equations={
        "density": "d(rho)/dt + div(rho * u) = s_rho",
        "velocity": (
            "rho * (du/dt + (u.grad)u) + grad(p) = div(tau(u)) + f_u, "
            "p = gas_constant * rho * temperature"
        ),
        "temperature": (
            "rho*c_v*(dT/dt + u.grad(T)) + p*div(u) = "
            "tau(u):grad(u) + thermal_conductivity*laplacian(T) + s_T"
        ),
    },
    parameters=[
        PDEParameter("gas_constant", "Ideal-gas constant relating pressure to rho*T"),
        PDEParameter("c_v", "Specific heat at constant volume"),
        PDEParameter("mu", "Dynamic shear viscosity"),
        PDEParameter("bulk_viscosity", "Bulk viscosity coefficient"),
        PDEParameter("thermal_conductivity", "Thermal conductivity coefficient"),
        PDEParameter("k", "Scalar-space degree; velocity uses k+1"),
    ],
    inputs={
        "density": InputSpec(
            name="density",
            shape="scalar",
            allow_source=True,
            allow_initial_condition=True,
        ),
        "velocity": InputSpec(
            name="velocity",
            shape="vector",
            allow_source=True,
            allow_initial_condition=True,
        ),
        "temperature": InputSpec(
            name="temperature",
            shape="scalar",
            allow_source=True,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "density": BoundaryFieldSpec(
            name="density",
            shape="scalar",
            operators=_DENSITY_BOUNDARY_OPERATORS,
            description=(
                "Boundary conditions for density. Smooth-flow v1 supports "
                "Dirichlet or periodic density boundaries."
            ),
        ),
        "velocity": BoundaryFieldSpec(
            name="velocity",
            shape="vector",
            operators=VECTOR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for velocity.",
        ),
        "temperature": BoundaryFieldSpec(
            name="temperature",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for temperature.",
        ),
    },
    states={
        "density": StateSpec(name="density", shape="scalar"),
        "velocity": StateSpec(name="velocity", shape="vector"),
        "temperature": StateSpec(name="temperature", shape="scalar"),
    },
    outputs={
        "density": OutputSpec(
            name="density",
            shape="scalar",
            output_mode="scalar",
            source_name="density",
        ),
        "velocity": OutputSpec(
            name="velocity",
            shape="vector",
            output_mode="components",
            source_name="velocity",
        ),
        "temperature": OutputSpec(
            name="temperature",
            shape="scalar",
            output_mode="scalar",
            source_name="temperature",
        ),
        "pressure": OutputSpec(
            name="pressure",
            shape="scalar",
            output_mode="scalar",
            source_name="pressure",
            source_kind="derived",
        ),
    },
    static_fields=[],
    steady_state=False,
    supported_dimensions=[2, 3],
)


def _positive_parameter(
    parameters: dict[str, float],
    name: str,
    *,
    allow_zero: bool = False,
) -> float:
    value = parameters[name]
    if allow_zero:
        if value < 0.0:
            raise ValueError(
                f"Preset 'compressible_navier_stokes' requires parameter '{name}' "
                f">= 0. Got {value}."
            )
    elif value <= 0.0:
        raise ValueError(
            f"Preset 'compressible_navier_stokes' requires parameter '{name}' "
            f"> 0. Got {value}."
        )
    return value


def _scalar_degree(parameters: dict[str, float]) -> int:
    raw_degree = parameters["k"]
    degree = int(raw_degree)
    if float(degree) != float(raw_degree) or degree < 1:
        raise ValueError(
            "Preset 'compressible_navier_stokes' requires integer parameter 'k' "
            f">= 1. Got {raw_degree}."
        )
    return degree


def _space_num_dofs(V: fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


def _stress_tensor(
    velocity,
    *,
    mu,
    bulk_viscosity,
    gdim: int,
):
    return 2.0 * mu * ufl.sym(ufl.grad(velocity)) + bulk_viscosity * ufl.div(
        velocity
    ) * ufl.Identity(gdim)


class _CompressibleNavierStokesProblem(TransientNonlinearProblem):
    supported_solver_strategies = (NONLINEAR_MIXED_DIRECT,)

    def validate_boundary_conditions(self, domain_geom):
        density_boundary_field = self.config.boundary_field("density")
        velocity_boundary_field = self.config.boundary_field("velocity")
        temperature_boundary_field = self.config.boundary_field("temperature")

        validate_boundary_field_structure(
            preset_name=self.spec.name,
            field_name="density",
            boundary_field=density_boundary_field,
            domain_geom=domain_geom,
            allowed_operators={"dirichlet", "periodic"},
        )
        validate_vector_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="velocity",
            boundary_field=velocity_boundary_field,
            domain_geom=domain_geom,
        )
        validate_scalar_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="temperature",
            boundary_field=temperature_boundary_field,
            domain_geom=domain_geom,
        )

        density_pairs = density_boundary_field.periodic_pair_keys()
        velocity_pairs = velocity_boundary_field.periodic_pair_keys()
        temperature_pairs = temperature_boundary_field.periodic_pair_keys()
        if density_pairs != velocity_pairs or density_pairs != temperature_pairs:
            raise ValueError(
                "Density, velocity, and temperature boundary conditions must use "
                "identical periodic side pairs."
            )

    def _update_output_views(self) -> None:
        self.density_out.x.array[:] = self.solution.x.array[self._density_dofs]
        self.density_out.x.scatter_forward()
        self.velocity_out.x.array[:] = self.solution.x.array[self._velocity_dofs]
        self.velocity_out.x.scatter_forward()
        self.temperature_out.x.array[:] = self.solution.x.array[self._temperature_dofs]
        self.temperature_out.x.scatter_forward()

    def _update_derived_outputs(self) -> None:
        self.pressure_out.interpolate(self._pressure_expr)
        self.pressure_out.x.scatter_forward()

    def _check_physical_admissibility(self, context: str) -> None:
        self._update_output_views()

        local_density = self.density_out.x.array
        local_temperature = self.temperature_out.x.array
        local_min_density = (
            float(np.min(local_density)) if local_density.size > 0 else float("inf")
        )
        local_min_temperature = (
            float(np.min(local_temperature))
            if local_temperature.size > 0
            else float("inf")
        )

        min_density = self.msh.comm.allreduce(local_min_density, op=MPI.MIN)
        min_temperature = self.msh.comm.allreduce(local_min_temperature, op=MPI.MIN)
        if min_density <= 0.0:
            raise ValueError(
                "Preset 'compressible_navier_stokes' produced non-positive density "
                f"during {context}. Minimum density: {min_density:.6g}."
            )
        if min_temperature <= 0.0:
            raise ValueError(
                "Preset 'compressible_navier_stokes' produced non-positive "
                f"temperature during {context}. Minimum temperature: "
                f"{min_temperature:.6g}."
            )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.domain_geom = domain_geom
        self.msh = domain_geom.mesh
        self.gdim = self.msh.geometry.dim

        parameters = self.config.parameters
        gas_constant = _positive_parameter(parameters, "gas_constant")
        c_v = _positive_parameter(parameters, "c_v")
        mu = _positive_parameter(parameters, "mu", allow_zero=True)
        bulk_viscosity = _positive_parameter(
            parameters,
            "bulk_viscosity",
            allow_zero=True,
        )
        thermal_conductivity = _positive_parameter(
            parameters,
            "thermal_conductivity",
            allow_zero=True,
        )
        scalar_degree = _scalar_degree(parameters)

        density_input = self.config.input("density")
        velocity_input = self.config.input("velocity")
        temperature_input = self.config.input("temperature")
        density_boundary_field = self.config.boundary_field("density")
        velocity_boundary_field = self.config.boundary_field("velocity")
        temperature_boundary_field = self.config.boundary_field("temperature")

        scalar_element = element(
            "Lagrange",
            self.msh.basix_cell(),
            scalar_degree,
            dtype=default_real_type,
        )
        vector_element = element(
            "Lagrange",
            self.msh.basix_cell(),
            scalar_degree + 1,
            shape=(self.gdim,),
            dtype=default_real_type,
        )
        mixed_space = fem.functionspace(
            self.msh,
            mixed_element([scalar_element, vector_element, scalar_element]),
        )

        bcs = [
            *apply_dirichlet_bcs_to_subspace(
                mixed_space.sub(0),
                domain_geom,
                density_boundary_field,
                parameters,
            ),
            *apply_vector_dirichlet_bcs_to_subspace(
                mixed_space.sub(1),
                domain_geom,
                velocity_boundary_field,
                parameters,
            ),
            *apply_dirichlet_bcs_to_subspace(
                mixed_space.sub(2),
                domain_geom,
                temperature_boundary_field,
                parameters,
            ),
        ]

        if density_boundary_field.has_periodic:
            mpc = self.create_periodic_constraint(
                mixed_space,
                domain_geom,
                density_boundary_field,
                bcs,
                constrained_spaces=[
                    mixed_space.sub(0),
                    mixed_space.sub(1),
                    mixed_space.sub(2),
                ],
            )
        else:
            mpc = None

        solution_space = mixed_space if mpc is None else mpc.function_space
        self.solution = fem.Function(solution_space)
        self.previous = fem.Function(solution_space)
        density_space, self._density_dofs = self.solution.function_space.sub(
            0
        ).collapse()
        velocity_space, self._velocity_dofs = self.solution.function_space.sub(
            1
        ).collapse()
        temperature_space, self._temperature_dofs = self.solution.function_space.sub(
            2
        ).collapse()
        self.density_out = fem.Function(density_space, name="density")
        self.velocity_out = fem.Function(velocity_space, name="velocity")
        self.temperature_out = fem.Function(temperature_space, name="temperature")
        pressure_space = fem.functionspace(
            self.msh,
            ("Discontinuous Lagrange", scalar_degree),
        )
        self.pressure_out = fem.Function(pressure_space, name="pressure")
        self._num_dofs = _space_num_dofs(self.solution.function_space)

        assert density_input.initial_condition is not None
        assert velocity_input.initial_condition is not None
        assert temperature_input.initial_condition is not None
        apply_ic(
            self.solution.sub(0),
            density_input.initial_condition,
            parameters,
            seed=self.config.seed,
        )
        apply_vector_ic(
            self.solution.sub(1),
            velocity_input.initial_condition,
            parameters,
            seed=self.config.seed,
        )
        apply_ic(
            self.solution.sub(2),
            temperature_input.initial_condition,
            parameters,
            seed=self.config.seed,
        )
        self.solution.x.scatter_forward()
        self._check_physical_admissibility("initialization")
        self.previous.x.array[:] = self.solution.x.array
        self.previous.x.scatter_forward()

        rho, velocity, temperature = ufl.split(self.solution)
        rho_prev, velocity_prev, temperature_prev = ufl.split(self.previous)
        test_rho, test_velocity, test_temperature = ufl.TestFunctions(mixed_space)
        trial = ufl.TrialFunction(mixed_space)

        gas_constant_c = fem.Constant(self.msh, default_real_type(gas_constant))
        c_v_c = fem.Constant(self.msh, default_real_type(c_v))
        mu_c = fem.Constant(self.msh, default_real_type(mu))
        bulk_viscosity_c = fem.Constant(
            self.msh,
            default_real_type(bulk_viscosity),
        )
        thermal_conductivity_c = fem.Constant(
            self.msh,
            default_real_type(thermal_conductivity),
        )
        delta_t = fem.Constant(self.msh, default_real_type(self.config.time.dt))
        zero_scalar = fem.Constant(self.msh, default_real_type(0.0))
        zero_vector = fem.Constant(
            self.msh,
            np.zeros(self.gdim, dtype=default_real_type),
        )
        assert density_input.source is not None
        density_source = build_ufl_field(
            self.msh,
            scalar_expression_to_config(density_input.source),
            parameters,
        )
        if density_source is None:
            density_source = zero_scalar

        assert velocity_input.source is not None
        velocity_source = build_vector_ufl_field(
            self.msh,
            velocity_input.source,
            parameters,
        )
        if velocity_source is None:
            velocity_source = zero_vector

        assert temperature_input.source is not None
        temperature_source = build_ufl_field(
            self.msh,
            scalar_expression_to_config(temperature_input.source),
            parameters,
        )
        if temperature_source is None:
            temperature_source = zero_scalar

        pressure = gas_constant_c * rho * temperature
        viscous_stress = _stress_tensor(
            velocity,
            mu=mu_c,
            bulk_viscosity=bulk_viscosity_c,
            gdim=self.gdim,
        )
        viscous_heating = ufl.inner(viscous_stress, ufl.grad(velocity))

        F_density = (
            ((rho - rho_prev) / delta_t) * test_rho * ufl.dx
            + ufl.div(rho * velocity) * test_rho * ufl.dx
            - density_source * test_rho * ufl.dx
        )

        F_velocity = (
            ufl.inner(rho * (velocity - velocity_prev) / delta_t, test_velocity)
            * ufl.dx
            + ufl.inner(
                rho * ufl.dot(ufl.grad(velocity), velocity),
                test_velocity,
            )
            * ufl.dx
            + ufl.inner(ufl.grad(pressure), test_velocity) * ufl.dx
            + ufl.inner(viscous_stress, ufl.grad(test_velocity)) * ufl.dx
            - ufl.inner(velocity_source, test_velocity) * ufl.dx
        )

        velocity_natural_form = build_vector_natural_bc_forms(
            test_velocity,
            domain_geom,
            velocity_boundary_field,
            parameters,
        )
        if velocity_natural_form is not None:
            F_velocity = F_velocity - velocity_natural_form

        F_temperature = (
            rho
            * c_v_c
            * (temperature - temperature_prev)
            / delta_t
            * test_temperature
            * ufl.dx
            + rho
            * c_v_c
            * ufl.inner(velocity, ufl.grad(temperature))
            * test_temperature
            * ufl.dx
            + pressure * ufl.div(velocity) * test_temperature * ufl.dx
            - viscous_heating * test_temperature * ufl.dx
            + thermal_conductivity_c
            * ufl.inner(ufl.grad(temperature), ufl.grad(test_temperature))
            * ufl.dx
            - temperature_source * test_temperature * ufl.dx
        )

        a_temp_bc, L_temp_bc = build_natural_bc_forms(
            temperature,
            test_temperature,
            domain_geom,
            temperature_boundary_field,
            parameters,
        )
        if a_temp_bc is not None:
            F_temperature = F_temperature + a_temp_bc
        if L_temp_bc is not None:
            F_temperature = F_temperature - L_temp_bc

        F = F_density + F_velocity + F_temperature
        J = ufl.derivative(F, self.solution, trial)

        self.problem = self.create_nonlinear_problem(
            F,
            self.solution,
            bcs=bcs,
            petsc_options_prefix="plm_compressible_navier_stokes_",
            J=J,
            mpc=mpc,
        )
        self._pressure_expr = fem.Expression(
            pressure,
            pressure_space.element.interpolation_points,
        )

    def step(self, t: float, dt: float) -> bool:
        self.previous.x.array[:] = self.solution.x.array
        self.previous.x.scatter_forward()
        self.problem.solve()
        converged = self.problem.solver.getConvergedReason() > 0
        if converged:
            self.solution.x.scatter_forward()
            self._check_physical_admissibility(f"t={t + dt:.6g}")
        return converged

    def get_output_fields(self) -> dict[str, fem.Function]:
        self._update_output_views()
        self._update_derived_outputs()
        return {
            "density": self.density_out,
            "velocity": self.velocity_out,
            "temperature": self.temperature_out,
            "pressure": self.pressure_out,
        }

    def get_num_dofs(self) -> int:
        return self._num_dofs


@register_preset("compressible_navier_stokes")
class CompressibleNavierStokesPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _COMPRESSIBLE_NAVIER_STOKES_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _CompressibleNavierStokesProblem(self.spec, config)

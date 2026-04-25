"""Thermal convection using the Boussinesq Rayleigh-Benard equations."""

import math

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem
from mpi4py import MPI

from plm_data.boundary_conditions.runtime import (
    apply_dirichlet_bcs_to_subspace,
    apply_vector_dirichlet_bcs_to_subspace,
    build_natural_bc_forms,
    build_vector_natural_bc_forms,
)
from plm_data.initial_conditions.runtime import apply_ic, apply_vector_ic
from plm_data.core.solver_strategies import TRANSIENT_MIXED_DIRECT
from plm_data.core.source_terms import build_source_form, build_vector_source_form
from plm_data.pdes.base import PDE, ProblemInstance, TransientLinearProblem
from plm_data.pdes.boundary_validation import (
    validate_scalar_standard_boundary_field,
    validate_vector_standard_boundary_field,
)
from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
    VECTOR_STANDARD_BOUNDARY_OPERATORS,
)
from plm_data.pdes.navier_stokes.pde import (
    _domain_average,
    _polynomial_degree,
    _positive_parameter,
    _space_num_dofs,
)


def _vertical_unit_vector(gdim: int):
    components = [ufl.as_ufl(0.0)] * gdim
    components[1] = ufl.as_ufl(1.0)
    return ufl.as_vector(components)


_THERMAL_CONVECTION_SPEC = PDESpec(
    name="thermal_convection",
    category="fluids",
    description="Thermal convection using the Boussinesq Rayleigh-Benard equations.",
    equations={
        "velocity": (
            "du/dt + (u_prev.grad)u = -grad(p) + sqrt(Pr/Ra)*laplacian(u) + T*e_y"
        ),
        "pressure": "div(u) = 0",
        "temperature": "dT/dt + u_prev.grad(T) = (1/sqrt(Ra*Pr))*laplacian(T) + f_T",
    },
    parameters=[
        PDEParameter("Ra", "Rayleigh number"),
        PDEParameter("Pr", "Prandtl number"),
        PDEParameter("k", "Polynomial degree parameter"),
    ],
    inputs={
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
        "velocity": BoundaryFieldSpec(
            name="velocity",
            shape="vector",
            operators=VECTOR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the velocity field.",
        ),
        "temperature": BoundaryFieldSpec(
            name="temperature",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the temperature field.",
        ),
    },
    states={
        "velocity": StateSpec(name="velocity", shape="vector"),
        "pressure": StateSpec(name="pressure", shape="scalar"),
        "temperature": StateSpec(name="temperature", shape="scalar"),
    },
    outputs={
        "velocity": OutputSpec(
            name="velocity",
            shape="vector",
            output_mode="components",
            source_name="velocity",
        ),
        "pressure": OutputSpec(
            name="pressure",
            shape="scalar",
            output_mode="scalar",
            source_name="pressure",
        ),
        "temperature": OutputSpec(
            name="temperature",
            shape="scalar",
            output_mode="scalar",
            source_name="temperature",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
)


class _ThermalConvectionProblem(TransientLinearProblem):
    supported_solver_strategies = (TRANSIENT_MIXED_DIRECT,)

    def validate_boundary_conditions(self, domain_geom):
        velocity_boundary = self.config.boundary_field("velocity")
        temperature_boundary = self.config.boundary_field("temperature")

        validate_vector_standard_boundary_field(
            pde_name=self.spec.name,
            field_name="velocity",
            boundary_field=velocity_boundary,
            domain_geom=domain_geom,
        )
        validate_scalar_standard_boundary_field(
            pde_name=self.spec.name,
            field_name="temperature",
            boundary_field=temperature_boundary,
            domain_geom=domain_geom,
        )
        if (
            velocity_boundary.periodic_pair_keys()
            != temperature_boundary.periodic_pair_keys()
        ):
            raise ValueError(
                "Velocity and temperature boundary conditions must use identical "
                "periodic side pairs."
            )

    def setup(self) -> None:
        assert self.config.time is not None

        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        gdim = self.msh.geometry.dim
        degree = _polynomial_degree(self.config.parameters, self.spec.name)
        rayleigh = _positive_parameter(self.config.parameters, "Ra", self.spec.name)
        prandtl = _positive_parameter(self.config.parameters, "Pr", self.spec.name)
        momentum_diffusivity = math.sqrt(prandtl / rayleigh)
        thermal_diffusivity = 1.0 / math.sqrt(rayleigh * prandtl)

        velocity_element = element(
            "Lagrange",
            self.msh.basix_cell(),
            degree + 1,
            shape=(gdim,),
            dtype=default_real_type,
        )
        pressure_element = element(
            "Lagrange",
            self.msh.basix_cell(),
            degree,
            dtype=default_real_type,
        )
        temperature_element = element(
            "Lagrange",
            self.msh.basix_cell(),
            degree,
            dtype=default_real_type,
        )
        mixed_space = fem.functionspace(
            self.msh,
            mixed_element([velocity_element, pressure_element, temperature_element]),
        )

        velocity_boundary = self.config.boundary_field("velocity")
        temperature_boundary = self.config.boundary_field("temperature")
        bcs = [
            *apply_vector_dirichlet_bcs_to_subspace(
                mixed_space.sub(0),
                domain_geom,
                velocity_boundary,
                self.config.parameters,
            ),
            *apply_dirichlet_bcs_to_subspace(
                mixed_space.sub(2),
                domain_geom,
                temperature_boundary,
                self.config.parameters,
            ),
        ]
        mpc = self.create_periodic_constraint(
            mixed_space,
            domain_geom,
            velocity_boundary,
            bcs,
            constrained_spaces=[
                mixed_space.sub(0),
                mixed_space.sub(1),
                mixed_space.sub(2),
            ],
        )
        solution_space = mixed_space if mpc is None else mpc.function_space

        self.solution = fem.Function(solution_space, name="solution")
        self.previous = fem.Function(solution_space, name="solution_prev")
        velocity_space, self._velocity_dofs = self.solution.function_space.sub(
            0
        ).collapse()
        pressure_space, self._pressure_dofs = self.solution.function_space.sub(
            1
        ).collapse()
        temperature_space, self._temperature_dofs = self.solution.function_space.sub(
            2
        ).collapse()
        self.velocity_out = fem.Function(velocity_space, name="velocity")
        self.pressure_out = fem.Function(pressure_space, name="pressure")
        self.temperature_out = fem.Function(temperature_space, name="temperature")
        self._num_dofs = _space_num_dofs(solution_space)

        velocity_input = self.config.input("velocity")
        temperature_input = self.config.input("temperature")
        assert velocity_input.initial_condition is not None
        assert temperature_input.initial_condition is not None
        apply_vector_ic(
            self.solution.sub(0),
            velocity_input.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        apply_ic(
            self.solution.sub(2),
            temperature_input.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        self.solution.x.scatter_forward()
        self.previous.x.array[:] = self.solution.x.array
        self.previous.x.scatter_forward()

        velocity, pressure, temperature = ufl.TrialFunctions(mixed_space)
        velocity_test, pressure_test, temperature_test = ufl.TestFunctions(mixed_space)
        velocity_prev, _, temperature_prev = ufl.split(self.previous)

        dt = fem.Constant(self.msh, default_real_type(self.config.time.dt))
        nu = fem.Constant(self.msh, default_real_type(momentum_diffusivity))
        kappa = fem.Constant(self.msh, default_real_type(thermal_diffusivity))
        vertical = _vertical_unit_vector(gdim)

        a = (
            ufl.inner(velocity, velocity_test) / dt * ufl.dx
            + ufl.inner(ufl.dot(ufl.grad(velocity), velocity_prev), velocity_test)
            * ufl.dx
            + nu * ufl.inner(ufl.grad(velocity), ufl.grad(velocity_test)) * ufl.dx
            - pressure * ufl.div(velocity_test) * ufl.dx
            + ufl.div(velocity) * pressure_test * ufl.dx
            - ufl.inner(temperature * vertical, velocity_test) * ufl.dx
            + ufl.inner(temperature, temperature_test) / dt * ufl.dx
            + ufl.inner(ufl.dot(ufl.grad(temperature), velocity_prev), temperature_test)
            * ufl.dx
            + kappa
            * ufl.inner(ufl.grad(temperature), ufl.grad(temperature_test))
            * ufl.dx
        )
        L = (
            ufl.inner(velocity_prev, velocity_test) / dt * ufl.dx
            + ufl.inner(temperature_prev, temperature_test) / dt * ufl.dx
        )

        assert velocity_input.source is not None
        velocity_source = build_vector_source_form(
            velocity_test,
            self.msh,
            velocity_input.source,
            self.config.parameters,
        )
        if velocity_source is not None:
            L = L + velocity_source

        assert temperature_input.source is not None
        temperature_source = build_source_form(
            temperature_test,
            self.msh,
            temperature_input.source,
            self.config.parameters,
        )
        if temperature_source is not None:
            L = L + temperature_source

        velocity_traction = build_vector_natural_bc_forms(
            velocity_test,
            domain_geom,
            velocity_boundary,
            self.config.parameters,
        )
        if velocity_traction is not None:
            L = L + velocity_traction

        a_bc, L_bc = build_natural_bc_forms(
            temperature,
            temperature_test,
            domain_geom,
            temperature_boundary,
            self.config.parameters,
        )
        if a_bc is not None:
            a = a + kappa * a_bc
        if L_bc is not None:
            L = L + kappa * L_bc

        self.problem = self.create_linear_problem(
            a,
            L,
            u=self.solution,
            bcs=bcs,
            petsc_options_prefix="plm_thermal_convection_",
            mpc=mpc,
        )

    def _update_output_views(self) -> None:
        self.velocity_out.x.array[:] = self.solution.x.array[self._velocity_dofs]
        self.velocity_out.x.scatter_forward()
        self.pressure_out.x.array[:] = self.solution.x.array[self._pressure_dofs]
        self.pressure_out.x.scatter_forward()
        self.temperature_out.x.array[:] = self.solution.x.array[self._temperature_dofs]
        self.temperature_out.x.scatter_forward()

    def _normalize_pressure(self) -> None:
        self._update_output_views()
        pressure_mean = _domain_average(self.msh, self.pressure_out)
        self.solution.x.array[self._pressure_dofs] -= pressure_mean
        self.solution.x.scatter_forward()

    def step(self, t: float, dt: float) -> bool:
        self.problem.solve()
        converged = self.problem.solver.getConvergedReason() > 0
        if converged:
            self.solution.x.scatter_forward()
            self._normalize_pressure()
            self.previous.x.array[:] = self.solution.x.array
            self.previous.x.scatter_forward()
        return converged

    def get_output_fields(self) -> dict[str, fem.Function]:
        self._update_output_views()
        return {
            "velocity": self.velocity_out,
            "pressure": self.pressure_out,
            "temperature": self.temperature_out,
        }

    def get_num_dofs(self) -> int:
        return int(self._num_dofs)

    def runtime_health_metrics(self, context: str) -> dict[str, float]:
        self._update_output_views()
        velocity_values = self.velocity_out.x.array
        pressure_values = self.pressure_out.x.array
        temperature_values = self.temperature_out.x.array
        for field_name, values in (
            ("velocity", velocity_values),
            ("pressure", pressure_values),
            ("temperature", temperature_values),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"PDE '{self.spec.name}' produced non-finite {field_name} "
                    f"values during {context}."
                )
        local_velocity_max = (
            float(np.max(np.abs(velocity_values))) if velocity_values.size else 0.0
        )
        local_pressure_max = (
            float(np.max(np.abs(pressure_values))) if pressure_values.size else 0.0
        )
        local_temperature_max = (
            float(np.max(np.abs(temperature_values)))
            if temperature_values.size
            else 0.0
        )
        return {
            "velocity_max_abs": self.msh.comm.allreduce(
                local_velocity_max,
                op=MPI.MAX,
            ),
            "pressure_max_abs": self.msh.comm.allreduce(
                local_pressure_max,
                op=MPI.MAX,
            ),
            "temperature_max_abs": self.msh.comm.allreduce(
                local_temperature_max,
                op=MPI.MAX,
            ),
        }


class ThermalConvectionPDE(PDE):
    @property
    def spec(self) -> PDESpec:
        return _THERMAL_CONVECTION_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _ThermalConvectionProblem(self.spec, config)

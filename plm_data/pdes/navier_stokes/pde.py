"""Incompressible Navier-Stokes PDE."""

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, fem
from mpi4py import MPI

from plm_data.boundary_conditions.runtime import (
    apply_vector_dirichlet_bcs_to_subspace,
    build_vector_natural_bc_forms,
)
from plm_data.initial_conditions.runtime import apply_vector_ic
from plm_data.core.solver_strategies import (
    TRANSIENT_MIXED_DIRECT,
    TRANSIENT_SADDLE_POINT,
)
from plm_data.fields.source_terms import build_vector_source_form
from plm_data.pdes.base import PDE, ProblemInstance, TransientLinearProblem
from plm_data.pdes.boundary_validation import validate_vector_standard_boundary_field
from plm_data.pdes.metadata import (
    PDESpec,
)
from plm_data.pdes.navier_stokes.spec import PDE_SPEC


def _space_num_dofs(V: fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


def _positive_parameter(
    parameters: dict[str, float], name: str, pde_name: str
) -> float:
    value = float(parameters[name])
    if value <= 0.0:
        raise ValueError(
            f"PDE '{pde_name}' requires parameter '{name}' > 0. Got {value}."
        )
    return value


def _polynomial_degree(parameters: dict[str, float], pde_name: str) -> int:
    raw_degree = parameters["k"]
    degree = int(raw_degree)
    if float(degree) != float(raw_degree) or degree < 1:
        raise ValueError(
            f"PDE '{pde_name}' requires integer parameter 'k' >= 1. Got {raw_degree}."
        )
    return degree


def _domain_average(msh, value) -> float:
    one = fem.Constant(msh, default_real_type(1.0))
    volume = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(one * ufl.dx)),
        op=MPI.SUM,
    )
    total = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(value * ufl.dx)),
        op=MPI.SUM,
    )
    return float(total / volume)


class _NavierStokesProblem(TransientLinearProblem):
    supported_solver_strategies = (TRANSIENT_SADDLE_POINT, TRANSIENT_MIXED_DIRECT)

    def validate_boundary_conditions(self, domain_geom):
        validate_vector_standard_boundary_field(
            pde_name=self.spec.name,
            field_name="velocity",
            boundary_field=self.config.boundary_field("velocity"),
            domain_geom=domain_geom,
        )

    def setup(self) -> None:
        assert self.config.time is not None

        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        gdim = self.msh.geometry.dim
        degree = _polynomial_degree(self.config.parameters, self.spec.name)
        reynolds = _positive_parameter(self.config.parameters, "Re", self.spec.name)
        viscosity = 1.0 / reynolds

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
        mixed_space = fem.functionspace(
            self.msh,
            mixed_element([velocity_element, pressure_element]),
        )

        velocity_boundary = self.config.boundary_field("velocity")
        bcs = apply_vector_dirichlet_bcs_to_subspace(
            mixed_space.sub(0),
            domain_geom,
            velocity_boundary,
            self.config.parameters,
        )
        mpc = self.create_periodic_constraint(
            mixed_space,
            domain_geom,
            velocity_boundary,
            bcs,
            constrained_spaces=[mixed_space.sub(0), mixed_space.sub(1)],
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
        self.velocity_out = fem.Function(velocity_space, name="velocity")
        self.pressure_out = fem.Function(pressure_space, name="pressure")
        self._num_dofs = _space_num_dofs(solution_space)

        velocity_input = self.config.input("velocity")
        assert velocity_input.initial_condition is not None
        apply_vector_ic(
            self.solution.sub(0),
            velocity_input.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        self.solution.x.scatter_forward()
        self.previous.x.array[:] = self.solution.x.array
        self.previous.x.scatter_forward()

        velocity, pressure = ufl.TrialFunctions(mixed_space)
        velocity_test, pressure_test = ufl.TestFunctions(mixed_space)
        velocity_prev, _ = ufl.split(self.previous)

        dt = fem.Constant(self.msh, default_real_type(self.config.time.dt))
        nu = fem.Constant(self.msh, default_real_type(viscosity))

        a = (
            ufl.inner(velocity, velocity_test) / dt * ufl.dx
            + ufl.inner(ufl.dot(ufl.grad(velocity), velocity_prev), velocity_test)
            * ufl.dx
            + nu * ufl.inner(ufl.grad(velocity), ufl.grad(velocity_test)) * ufl.dx
            - pressure * ufl.div(velocity_test) * ufl.dx
            + ufl.div(velocity) * pressure_test * ufl.dx
        )
        L = ufl.inner(velocity_prev, velocity_test) / dt * ufl.dx

        assert velocity_input.source is not None
        source_form = build_vector_source_form(
            velocity_test,
            self.msh,
            velocity_input.source,
            self.config.parameters,
        )
        if source_form is not None:
            L = L + source_form

        traction_form = build_vector_natural_bc_forms(
            velocity_test,
            domain_geom,
            velocity_boundary,
            self.config.parameters,
        )
        if traction_form is not None:
            L = L + traction_form

        self.problem = self.create_linear_problem(
            a,
            L,
            u=self.solution,
            bcs=bcs,
            petsc_options_prefix="plm_navier_stokes_",
            mpc=mpc,
        )

    def _update_output_views(self) -> None:
        self.velocity_out.x.array[:] = self.solution.x.array[self._velocity_dofs]
        self.velocity_out.x.scatter_forward()
        self.pressure_out.x.array[:] = self.solution.x.array[self._pressure_dofs]
        self.pressure_out.x.scatter_forward()

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
        return {"velocity": self.velocity_out, "pressure": self.pressure_out}

    def get_num_dofs(self) -> int:
        return int(self._num_dofs)

    def runtime_health_metrics(self, context: str) -> dict[str, float]:
        self._update_output_views()
        velocity_values = self.velocity_out.x.array
        pressure_values = self.pressure_out.x.array
        if not np.all(np.isfinite(velocity_values)):
            raise ValueError(
                f"PDE '{self.spec.name}' produced non-finite velocity values "
                f"during {context}."
            )
        if not np.all(np.isfinite(pressure_values)):
            raise ValueError(
                f"PDE '{self.spec.name}' produced non-finite pressure values "
                f"during {context}."
            )
        local_velocity_max = (
            float(np.max(np.abs(velocity_values))) if velocity_values.size else 0.0
        )
        local_pressure_max = (
            float(np.max(np.abs(pressure_values))) if pressure_values.size else 0.0
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
        }


class NavierStokesPDE(PDE):
    @property
    def spec(self) -> PDESpec:
        return PDE_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _NavierStokesProblem(self.spec, config)

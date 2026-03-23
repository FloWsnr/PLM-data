"""Base classes for PDE presets."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

from plm_data.core.config import SimulationConfig
from plm_data.core.mesh import DomainGeometry, create_domain
from plm_data.core.output import FrameWriter
from plm_data.presets.metadata import PDEMetadata


@dataclass
class RunResult:
    """Result of a simulation run. Filled by the preset after solving."""

    num_dofs: int
    solver_converged: bool
    wall_time: float = 0.0
    num_timesteps: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class PDEPreset(ABC):
    """Base class for all DOLFINx PDE presets.

    Each preset owns its entire solve logic. The `run()` method receives
    a config and a FrameWriter, and is responsible for mesh creation,
    function space setup, BC application, form assembly, solving, and
    writing output frames.
    """

    @property
    @abstractmethod
    def metadata(self) -> PDEMetadata:
        """Return metadata describing this PDE preset."""

    @abstractmethod
    def run(self, config: SimulationConfig, output: FrameWriter) -> RunResult:
        """Execute the full simulation."""


class SteadyLinearPreset(PDEPreset):
    """Convenience base for steady-state linear problems.

    Subclasses implement create_function_space(), create_boundary_conditions(),
    and create_forms(). The run() method handles domain creation, solving via
    LinearProblem, and writing a single output frame.
    """

    def create_domain(self, config: SimulationConfig) -> DomainGeometry:
        """Create the domain with tagged boundaries. Override for custom domains."""
        return create_domain(config.domain)

    @abstractmethod
    def create_function_space(
        self, domain_geom: DomainGeometry, config: SimulationConfig
    ) -> fem.FunctionSpace:
        """Create the FEM function space on the given mesh."""

    @abstractmethod
    def create_boundary_conditions(
        self,
        V: fem.FunctionSpace,
        domain_geom: DomainGeometry,
        config: SimulationConfig,
    ) -> list[fem.DirichletBC]:
        """Create boundary conditions."""

    @abstractmethod
    def create_forms(
        self,
        V: fem.FunctionSpace,
        domain_geom: DomainGeometry,
        config: SimulationConfig,
    ) -> tuple:
        """Return the bilinear form a and linear form L."""

    def run(self, config: SimulationConfig, output: FrameWriter) -> RunResult:
        domain_geom = self.create_domain(config)
        V = self.create_function_space(domain_geom, config)
        bcs = self.create_boundary_conditions(V, domain_geom, config)
        a, L = self.create_forms(V, domain_geom, config)

        problem = LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix="plm_",
            petsc_options=config.solver.options,
        )
        uh = problem.solve()

        converged = problem.solver.getConvergedReason() > 0
        if not converged:
            reason = problem.solver.getConvergedReason()
            raise RuntimeError(
                f"Steady linear solver did not converge (KSP reason={reason})"
            )
        output.write_frame({"u": uh}, t=0.0)  # type: ignore[reportArgumentType]

        return RunResult(
            num_dofs=V.dofmap.index_map.size_global,
            solver_converged=True,
        )


class TimeDependentPreset(PDEPreset):
    """Convenience base for time-dependent problems with a standard time loop.

    Subclasses implement setup() to create mesh, spaces, forms, solver, and
    initial condition. Then implement step() for a single timestep and
    get_output_fields() to return the current solution fields.
    """

    @abstractmethod
    def setup(self, config: SimulationConfig) -> None:
        """Create mesh, function spaces, forms, solver, and set initial condition.

        Store everything needed for time-stepping as instance attributes.
        Use create_domain() from plm_data.core.mesh to get a DomainGeometry.
        """

    @abstractmethod
    def step(self, t: float, dt: float) -> bool:
        """Advance the solution by one timestep.

        Returns:
            True if the solver converged, False otherwise.
        """

    @abstractmethod
    def get_output_fields(self) -> dict[str, fem.Function]:
        """Return current solution fields for output."""

    def get_num_dofs(self) -> int:
        """Return total number of DOFs."""
        return 0

    def run(self, config: SimulationConfig, output: FrameWriter) -> RunResult:
        self._solver_options = config.solver.options
        self.setup(config)

        assert config.dt is not None and config.t_end is not None
        dt = config.dt
        t_end = config.t_end
        num_frames = config.output.num_frames

        # Compute output times (evenly spaced including t=0 and t=t_end)
        if num_frames > 1:
            output_times = np.linspace(0.0, t_end, num_frames)
        else:
            output_times = np.array([t_end])
        next_output_idx = 0

        # Write initial condition
        if next_output_idx < len(output_times):
            output.write_frame(self.get_output_fields(), t=0.0)
            next_output_idx += 1

        t = 0.0
        num_steps = 0
        solver_converged = True
        while t < t_end - 1e-14 * dt:
            converged = self.step(t, dt)
            t += dt
            num_steps += 1

            if not converged:
                solver_converged = False
                raise RuntimeError(
                    f"Solver did not converge at t={t:.6g} (step {num_steps})"
                )

            if (
                next_output_idx < len(output_times)
                and t >= output_times[next_output_idx] - 1e-14 * dt
            ):
                output.write_frame(self.get_output_fields(), t=t)
                next_output_idx += 1

        return RunResult(
            num_dofs=self.get_num_dofs(),
            solver_converged=solver_converged,
            num_timesteps=num_steps,
        )

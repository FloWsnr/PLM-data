"""Base contracts for presets and reusable problem engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

from plm_data.core.config import SimulationConfig
from plm_data.core.logging import get_logger
from plm_data.core.mesh import DomainGeometry, create_domain
from plm_data.presets.metadata import PresetSpec

if TYPE_CHECKING:
    from plm_data.core.output import FrameWriter


@dataclass
class RunResult:
    """Result of a simulation run."""

    num_dofs: int
    solver_converged: bool
    wall_time: float = 0.0
    num_timesteps: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ProblemInstance(ABC):
    """Executable runtime object for a single simulation configuration."""

    def __init__(self, spec: PresetSpec, config: SimulationConfig):
        self.spec = spec
        self.config = config
        self._solver_options = config.solver.options

    @abstractmethod
    def run(self, output: FrameWriter) -> RunResult:
        """Execute the simulation and write output frames."""


class PDEPreset(ABC):
    """Factory for validated problem instances."""

    @property
    @abstractmethod
    def spec(self) -> PresetSpec:
        """Return the preset specification."""

    @abstractmethod
    def build_problem(self, config: SimulationConfig) -> ProblemInstance:
        """Build the runtime problem object for a validated config."""


class StationaryLinearProblem(ProblemInstance):
    """Reusable engine for steady-state linear problems."""

    def create_domain(self) -> DomainGeometry:
        """Create the domain with tagged boundaries."""
        return create_domain(self.config.domain)

    def count_dofs(self, V: fem.FunctionSpace) -> int:
        """Return the total scalar DOF count for the function space."""
        return V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    @abstractmethod
    def create_function_space(self, domain_geom: DomainGeometry) -> fem.FunctionSpace:
        """Create the FEM function space."""

    @abstractmethod
    def create_boundary_conditions(
        self,
        V: fem.FunctionSpace,
        domain_geom: DomainGeometry,
    ) -> list[fem.DirichletBC]:
        """Create strong boundary conditions."""

    @abstractmethod
    def create_forms(
        self,
        V: fem.FunctionSpace,
        domain_geom: DomainGeometry,
    ) -> tuple[Any, Any]:
        """Return the bilinear and linear forms."""

    @abstractmethod
    def export_solution_fields(self, solution: fem.Function) -> dict[str, fem.Function]:
        """Return base output fields after the solve."""

    def solver_prefix(self) -> str:
        """Return the PETSc option prefix for this problem."""
        return f"plm_{self.spec.name}_"

    def run(self, output: FrameWriter) -> RunResult:
        logger = get_logger("solver")
        domain_geom = self.create_domain()
        V = self.create_function_space(domain_geom)
        num_dofs = self.count_dofs(V)
        logger.info("  Solving stationary linear problem (%d DOFs)...", num_dofs)

        bcs = self.create_boundary_conditions(V, domain_geom)
        a, L = self.create_forms(V, domain_geom)

        problem = LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix=self.solver_prefix(),
            petsc_options=self._solver_options,
        )
        solution = problem.solve()

        reason = problem.solver.getConvergedReason()
        if reason <= 0:
            logger.error("  Solver did not converge (KSP reason=%s)", reason)
            raise RuntimeError(
                f"Stationary linear solver did not converge (KSP reason={reason})"
            )

        output.write_frame(self.export_solution_fields(solution), t=0.0)
        logger.info("  Solve complete (converged)")
        return RunResult(num_dofs=num_dofs, solver_converged=True)


class _TransientProblemBase(ProblemInstance):
    """Shared time-loop engine for transient problems."""

    @abstractmethod
    def setup(self) -> None:
        """Initialize spaces, forms, solvers, and state."""

    @abstractmethod
    def step(self, t: float, dt: float) -> bool:
        """Advance by one timestep and return convergence status."""

    @abstractmethod
    def get_output_fields(self) -> dict[str, fem.Function]:
        """Return base output fields for the current state."""

    def get_num_dofs(self) -> int:
        """Return the total number of degrees of freedom."""
        return 0

    def emit_initial_frame(self) -> bool:
        """Return whether to write the initial state at t=0."""
        return True

    def run(self, output: FrameWriter) -> RunResult:
        logger = get_logger("timestepper")
        self.setup()

        if self.config.time is None:
            raise ValueError(f"Preset '{self.spec.name}' requires a time section")

        dt = self.config.time.dt
        t_end = self.config.time.t_end
        num_frames = self.config.output.num_frames

        if num_frames > 1:
            output_times = np.linspace(0.0, t_end, num_frames)
        else:
            output_times = np.array([t_end])
        next_output_idx = 0

        total_steps = int(round(t_end / dt))
        log_every = max(1, total_steps // 10)
        logger.info(
            "  Time stepping: %d steps, dt=%s, t_end=%s", total_steps, dt, t_end
        )

        if self.emit_initial_frame() and next_output_idx < len(output_times):
            output.write_frame(self.get_output_fields(), t=0.0)
            next_output_idx += 1

        t = 0.0
        num_steps = 0
        while t < t_end - 1e-14 * dt:
            converged = self.step(t, dt)
            t += dt
            num_steps += 1

            if not converged:
                logger.error(
                    "  Solver did not converge at t=%.6g (step %d)", t, num_steps
                )
                raise RuntimeError(
                    f"Solver did not converge at t={t:.6g} (step {num_steps})"
                )

            if num_steps % log_every == 0 or num_steps == 1:
                progress = t / t_end * 100
                logger.info(
                    "  Step %d/%d (t=%.4g, %.0f%%)",
                    num_steps,
                    total_steps,
                    t,
                    progress,
                )

            if (
                next_output_idx < len(output_times)
                and t >= output_times[next_output_idx] - 1e-14 * dt
            ):
                output.write_frame(self.get_output_fields(), t=t)
                next_output_idx += 1

        logger.info("  Time stepping complete: %d steps", num_steps)
        return RunResult(
            num_dofs=self.get_num_dofs(),
            solver_converged=True,
            num_timesteps=num_steps,
        )


class TransientLinearProblem(_TransientProblemBase):
    """Reusable engine for transient linear problems."""


class TransientNonlinearProblem(_TransientProblemBase):
    """Reusable engine for transient nonlinear problems."""


class CustomProblem(ProblemInstance):
    """Escape hatch for problem families that need full control."""

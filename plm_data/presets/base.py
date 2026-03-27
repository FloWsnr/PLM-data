"""Base contracts for presets and reusable problem engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem as DolfinxLinearProblem
from dolfinx.fem.petsc import NonlinearProblem as DolfinxNonlinearProblem

from plm_data.core.config import BoundaryFieldConfig, SimulationConfig
from plm_data.core.logging import get_logger
from plm_data.core.mesh import DomainGeometry, create_domain
from plm_data.core.periodic import build_periodic_mpc, require_dolfinx_mpc
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

    def validate_boundary_conditions(self, domain_geom: DomainGeometry) -> None:
        """Validate boundary-condition semantics for this preset."""

    def load_domain_geometry(self) -> DomainGeometry:
        """Create and validate the domain geometry for this problem."""
        domain_geom = create_domain(self.config.domain)
        self.validate_boundary_conditions(domain_geom)
        return domain_geom

    def create_linear_problem(
        self,
        a,
        L,
        *,
        bcs: list[fem.DirichletBC] | None = None,
        petsc_options_prefix: str,
        petsc_options: dict[str, str] | None = None,
        u=None,
        mpc=None,
        kind=None,
        P=None,
    ):
        """Create a linear problem, switching to MPC when periodicity is active."""
        options = self._solver_options if petsc_options is None else petsc_options
        if mpc is None:
            kwargs: dict[str, Any] = {
                "bcs": [] if bcs is None else bcs,
                "petsc_options_prefix": petsc_options_prefix,
                "petsc_options": options,
            }
            if u is not None:
                kwargs["u"] = u
            if kind is not None:
                kwargs["kind"] = kind
            if P is not None:
                kwargs["P"] = P
            return DolfinxLinearProblem(a, L, **kwargs)

        dolfinx_mpc = require_dolfinx_mpc()
        kwargs = {
            "bcs": [] if bcs is None else bcs,
            "petsc_options_prefix": petsc_options_prefix,
            "petsc_options": options,
        }
        if u is not None:
            kwargs["u"] = u
        if P is not None:
            kwargs["P"] = P
        return dolfinx_mpc.LinearProblem(a, L, mpc, **kwargs)

    def create_periodic_constraint(
        self,
        V: fem.FunctionSpace,
        domain_geom: DomainGeometry,
        boundary_field: BoundaryFieldConfig,
        bcs: list[fem.DirichletBC] | None = None,
        constrained_spaces: list[fem.FunctionSpace] | None = None,
    ):
        """Build an MPC for the configured periodic domain, if any."""
        return build_periodic_mpc(
            V,
            domain_geom,
            boundary_field,
            bcs=bcs,
            constrained_spaces=constrained_spaces,
        )

    def create_nonlinear_problem(
        self,
        F,
        u,
        *,
        petsc_options_prefix: str,
        bcs: list[fem.DirichletBC] | None = None,
        J=None,
        P=None,
        kind=None,
        mpc=None,
    ):
        """Create a nonlinear problem, switching to MPC when periodicity is active."""
        if mpc is None:
            kwargs: dict[str, Any] = {
                "petsc_options_prefix": petsc_options_prefix,
                "bcs": [] if bcs is None else bcs,
                "petsc_options": self._solver_options,
            }
            if J is not None:
                kwargs["J"] = J
            if P is not None:
                kwargs["P"] = P
            if kind is not None:
                kwargs["kind"] = kind
            return DolfinxNonlinearProblem(F, u, **kwargs)

        dolfinx_mpc = require_dolfinx_mpc()
        kwargs = {
            "bcs": [] if bcs is None else bcs,
            "petsc_options": self._solver_options,
        }
        if J is not None:
            kwargs["J"] = J
        if P is not None:
            kwargs["P"] = P
        if kind is not None:
            kwargs["kind"] = kind
        return dolfinx_mpc.NonlinearProblem(F, u, mpc, **kwargs)

    @abstractmethod
    def run(self, output: "FrameWriter") -> RunResult:
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
        return self.load_domain_geometry()

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

    def periodic_boundary_field(self) -> BoundaryFieldConfig | None:
        """Return the BC field that drives periodic constraints, if any."""
        return None

    def solver_prefix(self) -> str:
        """Return the PETSc option prefix for this problem."""
        return f"plm_{self.spec.name}_"

    def run(self, output: "FrameWriter") -> RunResult:
        logger = get_logger("solver")
        domain_geom = self.create_domain()
        V = self.create_function_space(domain_geom)
        num_dofs = self.count_dofs(V)
        logger.info("  Solving stationary linear problem (%d DOFs)...", num_dofs)

        bcs = self.create_boundary_conditions(V, domain_geom)
        periodic_field = self.periodic_boundary_field()
        if periodic_field is None:
            mpc = None
        else:
            mpc = self.create_periodic_constraint(V, domain_geom, periodic_field, bcs)
        a, L = self.create_forms(V, domain_geom)

        problem = self.create_linear_problem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix=self.solver_prefix(),
            mpc=mpc,
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

    def run(self, output: "FrameWriter") -> RunResult:
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
        for num_steps in range(1, total_steps + 1):
            converged = self.step(t, dt)
            t = num_steps * dt

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

"""Simulation runner."""

from dataclasses import asdict, is_dataclass
import gc
import json
import logging
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
from mpi4py import MPI

from plm_data.core.runtime_config import SimulationConfig
from plm_data.core.health import combine_health_status
from plm_data.core.health import discover_solver_targets
from plm_data.core.logging import get_logger, setup_logging, teardown_logging
from plm_data.core.output import FrameWriter
from plm_data.core.runtime_realization import realize_simulation_config
from plm_data.pdes import get_pde


def _json_compatible(value):
    """Convert nested dataclasses and paths into JSON-compatible values."""

    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_compatible(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _reset_output_dir(output_dir: Path) -> None:
    """Remove and recreate one run output directory."""

    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    elif output_dir.exists():
        output_dir.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _delete_output_dir(output_dir: Path) -> None:
    """Delete one run output directory if it exists."""

    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    elif output_dir.exists():
        output_dir.unlink()


def _prepare_output_dir(
    output_dir: Path,
    *,
    reset: bool,
    comm: MPI.Intracomm | None = None,
) -> None:
    """Prepare one output directory once on rank 0 and broadcast failures."""

    active_comm = MPI.COMM_WORLD if comm is None else comm
    error_message: str | None = None

    if active_comm.rank == 0:
        try:
            if reset:
                _reset_output_dir(output_dir)
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"

    error_message = active_comm.bcast(error_message, root=0)
    if error_message is not None:
        raise RuntimeError(
            f"Failed to prepare output directory '{output_dir}': {error_message}"
        )


_PETSC_HANDLE_ATTRS = ("_solver", "_snes", "_A", "_b", "_x", "_P_mat")
RANDOM_ATTEMPT_BUDGET = 8


def _destroy_petsc_handles(target: object, *, destroyed_handle_ids: set[int]) -> None:
    """Destroy PETSc-backed handles on one wrapper object and clear references."""

    for attr_name in _PETSC_HANDLE_ATTRS:
        handle = getattr(target, attr_name, None)
        if handle is None:
            continue

        handle_id = id(handle)
        if handle_id not in destroyed_handle_ids:
            destroy = getattr(handle, "destroy", None)
            if callable(destroy):
                destroy()
            destroyed_handle_ids.add(handle_id)

        try:
            setattr(target, attr_name, None)
        except (AttributeError, TypeError):
            continue


def _cleanup_runtime_problem(problem: object | None) -> None:
    """Explicitly destroy PETSc solver objects before interpreter shutdown."""

    if problem is None or not hasattr(problem, "__dict__"):
        return

    destroyed_handle_ids: set[int] = set()
    targets = discover_solver_targets(problem.__dict__)
    for _, target in sorted(targets.items()):
        _destroy_petsc_handles(target, destroyed_handle_ids=destroyed_handle_ids)


class SimulationRunner:
    """Orchestrates a single simulation run."""

    def __init__(
        self,
        config: SimulationConfig,
        output_root: str | Path | None = None,
        *,
        output_dir_override: str | Path | None = None,
        sampled_info: dict[str, object] | None = None,
    ):
        if config.seed is None:
            raise ValueError(
                "Simulation runs require an explicit seed from the config or an "
                "explicit seed override."
            )
        self.config = realize_simulation_config(config)
        self.pde = get_pde(self.config.pde)
        self.output_root = (
            Path(output_root) if output_root is not None else self.config.output.path
        )
        if self.output_root is None:
            raise ValueError("SimulationRunner requires an output root directory.")
        self.output_dir_override = (
            Path(output_dir_override) if output_dir_override is not None else None
        )
        self.sampled_info = {} if sampled_info is None else dict(sampled_info)

    def _serialize_config(self) -> dict:
        """Return the resolved simulation config in JSON-compatible form."""

        return {
            "output_root": str(self.output_root),
            "resolved": _json_compatible(self.config),
        }

    def _build_run_summary(
        self,
        *,
        status: str,
        stage: str,
        result,
        output: FrameWriter | None,
        error: Exception | None,
    ) -> dict[str, object]:
        """Return a compact summary of the current run state."""

        output_health_status = "pass"
        if output is not None:
            output_health_status = output.health_summary()["status"]

        solver_health_status = "pass"
        runtime_health_status = "pass"
        if result is not None:
            solver_health_status = result.diagnostics.get("solver_health", {}).get(
                "status",
                "pass",
            )
            runtime_health_status = result.diagnostics.get("runtime_health", {}).get(
                "status",
                "pass",
            )

        if status == "failed":
            health_status = "fail"
        elif status == "running":
            health_status = None
        else:
            health_status = combine_health_status(
                output_health_status,
                solver_health_status,
                runtime_health_status,
            )

        return {
            "status": status,
            "stage": stage,
            "health_status": health_status,
            "output_health_status": output_health_status
            if output is not None
            else None,
            "solver_health_status": (
                solver_health_status if result is not None else None
            ),
            "runtime_health_status": (
                runtime_health_status if result is not None else None
            ),
            "solver_converged": None if result is None else result.solver_converged,
            "num_timesteps": None if result is None else result.num_timesteps,
            "num_frames": 0 if output is None else output.frame_count,
            "error_type": None if error is None else type(error).__name__,
            "error_message": None if error is None else str(error),
        }

    def _write_run_metadata(
        self,
        output_dir: Path,
        *,
        status: str,
        stage: str,
        result=None,
        output: FrameWriter | None = None,
        error: Exception | None = None,
        total_wall_seconds: float | None = None,
        problem_run_seconds: float = 0.0,
        output_finalize_call_seconds: float = 0.0,
    ) -> None:
        """Write run-level metadata for successful or failed runs."""

        if MPI.COMM_WORLD.rank != 0:
            return

        timings = {
            "problem_run_seconds": problem_run_seconds,
            "output_finalize_call_seconds": output_finalize_call_seconds,
            "total_wall_seconds": total_wall_seconds,
            "output": output.timing_summary() if output is not None else None,
        }
        diagnostics = None if result is None else result.diagnostics
        run_meta = {
            "status": status,
            "stage": stage,
            "pde": self.pde.spec.name,
            "category": self.pde.spec.category,
            "domain": self.config.domain.type,
            "seed": self.config.seed,
            "run_id": self.sampled_info.get("run_id"),
            "output_dir": str(output_dir),
            "log_path": str(output_dir / "simulation.log"),
            "num_dofs": None if result is None else result.num_dofs,
            "solver_converged": None if result is None else result.solver_converged,
            "num_timesteps": None if result is None else result.num_timesteps,
            "num_frames": 0 if output is None else output.frame_count,
            "timings": timings,
            "summary": self._build_run_summary(
                status=status,
                stage=stage,
                result=result,
                output=output,
                error=error,
            ),
            "config": self._serialize_config(),
            "random_sampling": self.sampled_info or None,
            "diagnostics": diagnostics,
            "frames_meta_path": (
                str(output_dir / "frames_meta.json")
                if status == "success" and (output_dir / "frames_meta.json").exists()
                else None
            ),
        }
        if error is not None:
            run_meta["error"] = {
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ),
            }
        else:
            run_meta["error"] = None

        with open(output_dir / "run_meta.json", "w") as f:
            json.dump(run_meta, f, indent=2)

    def resolve_output_dir(self, output_subdir: str | None = None) -> Path:
        """Return the target directory for this run."""

        if self.output_dir_override is None:
            output_dir = self.output_root / self.pde.spec.category / self.pde.spec.name
        else:
            output_dir = self.output_dir_override
        if output_subdir is not None:
            output_dir = output_dir / output_subdir
        return output_dir

    def run(
        self,
        console_level: int = logging.INFO,
        *,
        output_subdir: str | None = None,
        cleanup_output_dir: bool = True,
    ) -> dict:
        spec = self.pde.spec
        output_dir = self.resolve_output_dir(output_subdir)
        problem = None
        if cleanup_output_dir:
            _prepare_output_dir(output_dir, reset=True)
        else:
            _prepare_output_dir(output_dir, reset=False)

        logger = get_logger("runner")
        output: FrameWriter | None = None
        result = None
        problem_run_time = 0.0
        finalize_time = 0.0
        run_start = time.perf_counter()
        stage = "setup"
        self._write_run_metadata(
            output_dir,
            status="running",
            stage=stage,
            result=None,
            output=None,
            total_wall_seconds=0.0,
        )

        try:
            setup_logging(output_dir, console_level=console_level)
            logger.info("Running '%s' (%s)...", spec.name, spec.category)
            logger.info(
                "  Domain: %s, Output: %s",
                self.config.domain.type,
                self.config.output.resolution,
            )
            if self.config.time is not None:
                logger.info(
                    "  Time: 0 → %s, dt=%s",
                    self.config.time.t_end,
                    self.config.time.dt,
                )
            logger.debug("  Seed: %s", self.config.seed)
            logger.info(
                "  Solver: strategy=%s, profile=%s",
                self.config.solver.strategy,
                self.config.solver.profile_name,
            )
            logger.debug("  Solver options: %s", self.config.solver.options)

            if self.config.seed is not None:
                np.random.seed(self.config.seed)

            stage = "build_output"
            output = FrameWriter(output_dir, self.config, spec)
            stage = "build_problem"
            problem = self.pde.build_problem(self.config)

            stage = "problem_run"
            t_start = time.perf_counter()
            result = problem.run(output)
            problem_run_time = time.perf_counter() - t_start

            stage = "output_finalize"
            finalize_start = time.perf_counter()
            output.finalize(run_diagnostics=result.diagnostics)
            finalize_time = time.perf_counter() - finalize_start

            result.wall_time = problem_run_time + finalize_time
            result.diagnostics["timings"] = {
                "problem_run_seconds": problem_run_time,
                "output_finalize_call_seconds": finalize_time,
                "total_wall_seconds": result.wall_time,
                "output": output.timing_summary(),
            }
            self._write_run_metadata(
                output_dir,
                status="success",
                stage="completed",
                result=result,
                output=output,
                total_wall_seconds=result.wall_time,
                problem_run_seconds=problem_run_time,
                output_finalize_call_seconds=finalize_time,
            )

            logger.info("  Converged: %s", result.solver_converged)
            logger.info("  DOFs: %s", result.num_dofs)
            logger.info("  Frames: %s", output.frame_count)
            logger.info("  Problem run: %.2fs", problem_run_time)
            logger.info("  Output finalize: %.2fs", finalize_time)
            logger.info("  Wall time: %.2fs", result.wall_time)
            logger.info("  Output: %s", output_dir)
            logger.info("  Log: %s", output_dir / "simulation.log")
        except Exception as exc:
            total_wall = time.perf_counter() - run_start
            if result is not None:
                result.wall_time = total_wall
                result.diagnostics.setdefault(
                    "timings",
                    {
                        "problem_run_seconds": problem_run_time,
                        "output_finalize_call_seconds": finalize_time,
                        "total_wall_seconds": total_wall,
                        "output": (None if output is None else output.timing_summary()),
                    },
                )
            try:
                self._write_run_metadata(
                    output_dir,
                    status="failed",
                    stage=stage,
                    result=result,
                    output=output,
                    error=exc,
                    total_wall_seconds=total_wall,
                    problem_run_seconds=problem_run_time,
                    output_finalize_call_seconds=finalize_time,
                )
            except Exception:
                logger.exception("  Failed to write run_meta.json")
            raise
        finally:
            _cleanup_runtime_problem(problem)
            problem = None
            gc.collect()
            teardown_logging()

        return {
            "pde": spec.name,
            "category": spec.category,
            "domain": self.config.domain.type,
            "seed": self.config.seed,
            "run_id": self.sampled_info.get("run_id"),
            "wall_time": result.wall_time,
            "num_dofs": result.num_dofs,
            "solver_converged": result.solver_converged,
            "num_timesteps": result.num_timesteps,
            "num_frames": output.frame_count,
            "timings": result.diagnostics["timings"],
            "output_dir": str(output_dir),
        }


def _sample_random_config_on_rank_zero(seed: int, attempt: int):
    """Sample on rank 0 and broadcast the result to every MPI rank."""

    from plm_data.sampling import sample_random_runtime_config

    comm = MPI.COMM_WORLD
    payload: dict[str, object] | None = None
    if comm.rank == 0:
        try:
            payload = {
                "sampled": sample_random_runtime_config(seed=seed, attempt=attempt),
                "error": None,
            }
        except Exception as exc:
            payload = {
                "sampled": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    payload = comm.bcast(payload, root=0)
    if payload is None:
        raise RuntimeError("Random sampling broadcast failed.")
    if payload["error"] is not None:
        raise RuntimeError(f"Random sampling failed: {payload['error']}")
    return payload["sampled"]


def _cleanup_failed_random_output(output_dir: Path) -> None:
    """Remove failed random-attempt output once all ranks leave the runner."""

    comm = MPI.COMM_WORLD
    comm.barrier()
    if comm.rank == 0:
        _delete_output_dir(output_dir)
    comm.barrier()


def run_random_simulation(
    *,
    seed: int,
    output_root: str | Path = "./output",
    console_level: int = logging.INFO,
    attempt_budget: int = RANDOM_ATTEMPT_BUDGET,
) -> dict[str, object]:
    """Run one bounded-retry random simulation from an explicit seed."""

    if attempt_budget < 1:
        raise ValueError("attempt_budget must be at least 1.")

    failures: list[str] = []
    output_root = Path(output_root)

    for attempt in range(attempt_budget):
        sampled = _sample_random_config_on_rank_zero(seed, attempt)
        output_dir = sampled.output_dir(output_root)
        sampled_info = sampled.metadata()
        try:
            runner = SimulationRunner(
                sampled.config,
                output_root,
                output_dir_override=output_dir,
                sampled_info=sampled_info,
            )
            summary = runner.run(console_level=console_level)
            summary["random_sampling"] = sampled_info
            return summary
        except Exception as exc:
            failures.append(
                f"attempt {attempt}: {sampled.pde_name}/{sampled.domain_name} "
                f"failed with {type(exc).__name__}: {exc}"
            )
            _cleanup_failed_random_output(output_dir)

    failure_summary = "; ".join(failures[-3:]) if failures else "no attempts ran"
    raise RuntimeError(
        f"Random simulation failed after {attempt_budget} attempts. {failure_summary}"
    )

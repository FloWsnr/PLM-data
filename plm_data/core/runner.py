"""Simulation runner."""

from dataclasses import asdict, is_dataclass, replace
import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np
from mpi4py import MPI

from plm_data.core.config import SimulationConfig, load_config
from plm_data.core.health import combine_health_status
from plm_data.core.logging import get_logger, setup_logging, teardown_logging
from plm_data.core.output import FrameWriter
from plm_data.presets import get_preset


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


class SimulationRunner:
    """Orchestrates a single simulation run."""

    def __init__(
        self,
        config: SimulationConfig,
        output_root: str | Path | None = None,
        *,
        config_source: str | Path | None = None,
    ):
        if config.seed is None:
            raise ValueError(
                "Simulation runs require an explicit seed from the config or '--seed'."
            )
        self.config = config
        self.preset = get_preset(config.preset)
        self.config_source = (
            Path(config_source).resolve() if config_source is not None else None
        )
        self.output_root = (
            Path(output_root) if output_root is not None else self.config.output.path
        )
        if self.output_root is None:
            raise ValueError("SimulationRunner requires an output root directory.")

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        output_root: str | Path,
        *,
        seed: int | None = None,
    ) -> "SimulationRunner":
        config = load_config(path)
        if seed is not None:
            config = replace(config, seed=seed)
        return cls(config, output_root, config_source=path)

    def _serialize_config(self) -> dict:
        """Return the resolved simulation config in JSON-compatible form."""

        return {
            "source_path": None
            if self.config_source is None
            else str(self.config_source),
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
            "preset": self.preset.spec.name,
            "category": self.preset.spec.category,
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

    def run(self, console_level: int = logging.INFO) -> dict:
        spec = self.preset.spec
        output_dir = self.output_root / spec.category / spec.name
        output_dir.mkdir(parents=True, exist_ok=True)

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
            problem = self.preset.build_problem(self.config)

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
            teardown_logging()

        return {
            "preset": spec.name,
            "category": spec.category,
            "wall_time": result.wall_time,
            "num_dofs": result.num_dofs,
            "solver_converged": result.solver_converged,
            "num_timesteps": result.num_timesteps,
            "num_frames": output.frame_count,
            "timings": result.diagnostics["timings"],
            "output_dir": str(output_dir),
        }

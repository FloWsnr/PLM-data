"""Simulation runner."""

import logging
import time
from pathlib import Path

import numpy as np

from plm_data.core.config import SimulationConfig, load_config
from plm_data.core.logging import get_logger, setup_logging, teardown_logging
from plm_data.core.output import FrameWriter
from plm_data.presets import get_preset


class SimulationRunner:
    """Orchestrates a single simulation run."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.preset = get_preset(config.preset)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimulationRunner":
        return cls(load_config(path))

    def run(self, console_level: int = logging.INFO) -> dict:
        spec = self.preset.spec
        output_dir = Path(self.config.output.path) / spec.category / spec.name
        output_dir.mkdir(parents=True, exist_ok=True)

        setup_logging(output_dir, console_level=console_level)
        logger = get_logger("runner")

        try:
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
            logger.debug("  Solver options: %s", self.config.solver.options)

            if self.config.seed is not None:
                np.random.seed(self.config.seed)

            output = FrameWriter(output_dir, self.config, spec)
            problem = self.preset.build_problem(self.config)

            t_start = time.perf_counter()
            result = problem.run(output)
            problem_run_time = time.perf_counter() - t_start

            finalize_start = time.perf_counter()
            output.finalize()
            finalize_time = time.perf_counter() - finalize_start

            result.wall_time = problem_run_time + finalize_time
            result.diagnostics["timings"] = {
                "problem_run_seconds": problem_run_time,
                "output_finalize_call_seconds": finalize_time,
                "total_wall_seconds": result.wall_time,
                "output": output.timing_summary(),
            }

            logger.info("  Converged: %s", result.solver_converged)
            logger.info("  DOFs: %s", result.num_dofs)
            logger.info("  Frames: %s", output.frame_count)
            logger.info("  Problem run: %.2fs", problem_run_time)
            logger.info("  Output finalize: %.2fs", finalize_time)
            logger.info("  Wall time: %.2fs", result.wall_time)
            logger.info("  Output: %s", output_dir)
            logger.info("  Log: %s", output_dir / "simulation.log")
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

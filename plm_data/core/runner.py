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
        meta = self.preset.metadata
        output_dir = Path(self.config.output.path) / meta.category / meta.name
        output_dir.mkdir(parents=True, exist_ok=True)

        setup_logging(output_dir, console_level=console_level)
        logger = get_logger("runner")

        try:
            logger.info("Running '%s' (%s)...", meta.name, meta.category)
            logger.info(
                "  Domain: %s, Output: %s",
                self.config.domain.type,
                self.config.output_resolution,
            )
            if self.config.t_end is not None:
                logger.info("  Time: 0 → %s, dt=%s", self.config.t_end, self.config.dt)
            logger.debug("  Seed: %s", self.config.seed)
            logger.debug("  Solver options: %s", self.config.solver.options)

            if self.config.seed is not None:
                np.random.seed(self.config.seed)

            output = FrameWriter(output_dir, self.config)

            t_start = time.perf_counter()
            result = self.preset.run(self.config, output)
            result.wall_time = time.perf_counter() - t_start

            output.finalize()

            logger.info("  Converged: %s", result.solver_converged)
            logger.info("  DOFs: %s", result.num_dofs)
            logger.info("  Frames: %s", output.frame_count)
            logger.info("  Wall time: %.2fs", result.wall_time)
            logger.info("  Output: %s", output_dir)
            logger.info("  Log: %s", output_dir / "simulation.log")
        finally:
            teardown_logging()

        return {
            "preset": meta.name,
            "category": meta.category,
            "wall_time": result.wall_time,
            "num_dofs": result.num_dofs,
            "solver_converged": result.solver_converged,
            "num_timesteps": result.num_timesteps,
            "num_frames": output.frame_count,
            "output_dir": str(output_dir),
        }

"""Simulation runner."""

import time
from pathlib import Path

import numpy as np

from plm_data.core.config import SimulationConfig, load_config
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

    def run(self, verbose: bool = True) -> dict:
        meta = self.preset.metadata
        output_dir = Path(self.config.output.path) / meta.category / meta.name
        output_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"Running '{meta.name}' ({meta.category})...")
            print(f"  Domain: {self.config.domain.type}, Output: {self.config.output_resolution}")
            if self.config.t_end is not None:
                print(f"  Time: 0 → {self.config.t_end}, dt={self.config.dt}")

        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        output = FrameWriter(output_dir, self.config)

        t_start = time.perf_counter()
        result = self.preset.run(self.config, output)
        result.wall_time = time.perf_counter() - t_start

        output.finalize()

        if verbose:
            print(f"  Converged: {result.solver_converged}")
            print(f"  DOFs: {result.num_dofs}")
            print(f"  Frames: {output.frame_count}")
            print(f"  Wall time: {result.wall_time:.2f}s")
            print(f"  Output: {output_dir}")

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

"""Output handlers for simulation data."""

import json
from pathlib import Path

import numpy as np
from dolfinx import fem

from plm_data.core.config import SimulationConfig
from plm_data.core.interpolation import function_to_array


class FrameWriter:
    """Captures field snapshots during simulation.

    Presets call write_frame() to emit snapshots. The writer handles
    interpolation to regular grids and saving in configured formats.
    """

    def __init__(self, output_dir: Path, config: SimulationConfig):
        self.output_dir = output_dir
        self.config = config
        self.frame_count = 0
        self.frame_times: list[float] = []
        self.field_names: list[str] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_frame(self, fields: dict[str, fem.Function], t: float):
        """Capture a snapshot of one or more fields.

        Args:
            fields: Map from field name to DOLFINx Function.
            t: Current simulation time.
        """
        if not self.field_names:
            self.field_names = list(fields.keys())

        res = tuple(self.config.output_resolution)

        for name, func in fields.items():
            arr = function_to_array(func, resolution=res)

            if "numpy" in self.config.output.formats:
                frame_dir = self.output_dir / "frames" / name
                frame_dir.mkdir(parents=True, exist_ok=True)
                np.save(frame_dir / f"{self.frame_count:06d}.npy", arr)

        self.frame_times.append(t)
        self.frame_count += 1

    def finalize(self):
        """Write summary metadata after all frames are captured."""
        metadata = {
            "num_frames": self.frame_count,
            "times": self.frame_times,
            "field_names": self.field_names,
            "output_resolution": self.config.output_resolution,
            "domain_type": self.config.domain.type,
            "domain_params": self.config.domain.params,
        }
        with open(self.output_dir / "frames_meta.json", "w") as f:
            json.dump(metadata, f, indent=2)

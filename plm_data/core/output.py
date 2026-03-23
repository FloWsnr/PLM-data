"""Output handlers for simulation data."""

import json
from pathlib import Path

import numpy as np
from dolfinx import fem

from plm_data.core.config import SimulationConfig
from plm_data.core.interpolation import function_to_array
from plm_data.core.logging import get_logger


class FrameWriter:
    """Captures field snapshots during simulation.

    Presets call write_frame() to emit snapshots. The writer handles
    interpolation to regular grids and saving in configured formats.
    All frames are accumulated in memory and saved as a single numpy
    array per field in finalize().
    """

    def __init__(self, output_dir: Path, config: SimulationConfig):
        self.output_dir = output_dir
        self.config = config
        self.frame_count = 0
        self.frame_times: list[float] = []
        self.field_names: list[str] = []
        self._field_frames: dict[str, list[np.ndarray]] = {}
        self._logger = get_logger("output")

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
            if name not in self._field_frames:
                self._field_frames[name] = []
            self._field_frames[name].append(arr)

        self.frame_times.append(t)
        self.frame_count += 1
        self._logger.debug("  Frame %d captured at t=%.6g", self.frame_count, t)

    def finalize(self):
        """Stack all frames per field into a single array and save."""
        if "numpy" in self.config.output.formats:
            for name, frames in self._field_frames.items():
                stacked = np.stack(frames, axis=0)
                np.save(self.output_dir / f"{name}.npy", stacked)
            self._logger.info(
                "  Saved %d frames (%s) to %s",
                self.frame_count,
                ", ".join(self.field_names),
                self.output_dir,
            )

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

"""GIF animation output format."""

import importlib
from pathlib import Path

import numpy as np
from mpi4py import MPI

from plm_data.core.formats._rendering import render_animation, render_scalar_gif_fast
from plm_data.core.logging import get_logger


class GifWriter:
    """Accumulates interpolated frames and renders animated GIFs."""

    def __init__(self, output_dir: Path):
        try:
            importlib.import_module("matplotlib")
        except ImportError:
            raise ImportError(
                "GIF output format requires matplotlib: pip install matplotlib"
            ) from None

        self._output_dir = output_dir
        self._field_frames: dict[str, list[np.ndarray]] = {}
        self._logger = get_logger("output.gif")
        self._rank = MPI.COMM_WORLD.rank

    def on_frame_field(self, name: str, arr: np.ndarray, t: float) -> None:
        """Accumulate one interpolated field array."""
        if self._rank == 0:
            self._field_frames.setdefault(name, []).append(arr)

    def finalize(self) -> None:
        """Render accumulated frames as animated GIFs."""
        if self._rank != 0:
            return

        for name, frames in self._field_frames.items():
            output_path = self._output_dir / f"{name}.gif"
            first_frame = frames[0]
            if np.asarray(first_frame).ndim == 2:
                render_scalar_gif_fast(frames, output_path)
            else:
                stacked = np.stack(frames, axis=0)
                render_animation(stacked, name, output_path, writer_name="pillow")
            self._logger.info("  Saved GIF: %s", output_path)

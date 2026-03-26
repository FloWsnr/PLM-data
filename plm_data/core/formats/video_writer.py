"""MP4 video output format."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from mpi4py import MPI

from plm_data.core.formats._rendering import render_animation
from plm_data.core.logging import get_logger


class VideoWriter:
    """Accumulates interpolated frames and renders MP4 videos."""

    def __init__(self, output_dir: Path):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            raise ImportError(
                "Video output format requires matplotlib: pip install matplotlib"
            ) from None

        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "Video output format requires ffmpeg to be installed and on PATH"
            )

        self._output_dir = output_dir
        self._field_frames: dict[str, list[np.ndarray]] = {}
        self._logger = get_logger("output.video")
        self._rank = MPI.COMM_WORLD.rank

    def on_frame_field(self, name: str, arr: np.ndarray, t: float) -> None:
        """Accumulate one interpolated field array."""
        if self._rank == 0:
            self._field_frames.setdefault(name, []).append(arr)

    def finalize(self) -> None:
        """Render accumulated frames as MP4 videos."""
        if self._rank != 0:
            return

        for name, frames in self._field_frames.items():
            stacked = np.stack(frames, axis=0)
            output_path = self._output_dir / f"{name}.mp4"
            render_animation(stacked, name, output_path, writer_name="ffmpeg")
            self._logger.info("  Saved video: %s", output_path)

"""Numpy array output format."""

from pathlib import Path

import numpy as np

from plm_data.core.logging import get_logger


class NumpyWriter:
    """Accumulates interpolated frames and saves as .npy arrays."""

    def __init__(self, output_dir: Path, field_names: list[str]):
        self._output_dir = output_dir
        self._field_names = field_names
        self._field_frames: dict[str, list[np.ndarray]] = {}
        self._logger = get_logger("output.numpy")

    def on_frame_field(self, name: str, arr: np.ndarray, t: float) -> None:
        """Accumulate one interpolated field array."""
        self._field_frames.setdefault(name, []).append(arr)

    def finalize(self) -> None:
        """Stack all frames per field and save as .npy files."""
        for name, frames in self._field_frames.items():
            np.save(self._output_dir / f"{name}.npy", np.stack(frames, axis=0))
        self._logger.info(
            "  Saved %d numpy arrays to %s",
            len(self._field_frames),
            self._output_dir,
        )

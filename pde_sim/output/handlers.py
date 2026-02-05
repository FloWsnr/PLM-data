"""Concrete output format handlers.

Handlers implement a small interface so the OutputManager can write multiple formats
simultaneously (e.g. png + numpy) without duplicating orchestration logic.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap

from .render import render_colormap_rgb


class OutputHandler(ABC):
    """Abstract base class for output format handlers."""

    @abstractmethod
    def initialize(
        self,
        output_dir: Path,
        field_configs: list[tuple[str, str]],
        dpi: int,
        figsize: tuple[int, int],
        *,
        expected_num_frames: int | None = None,
    ) -> None:
        """Set up output directories/files."""

    @abstractmethod
    def save_frame(
        self,
        data: np.ndarray,
        field_name: str,
        frame_index: int,
        simulation_time: float,
        vmin: float | None,
        vmax: float | None,
        colormap: str,
    ) -> None:
        """Save a single frame for one field."""

    @abstractmethod
    def finalize(self) -> dict[str, Any]:
        """Complete output and return format-specific metadata."""


class Output1DHandler(OutputHandler):
    """Handles 1D output as space-time diagrams and optional animations."""

    def __init__(self, output_format: str = "png", fps: int = 30) -> None:
        self.output_format = output_format
        self.fps = fps
        self.output_dir: Path | None = None
        self.spacetime_data: dict[str, list[np.ndarray]] = {}
        self.times: list[float] = []
        self.field_colormaps: dict[str, str] = {}
        self.dpi: int = 128
        self.figsize: tuple[int, int] = (10, 8)
        self._times_recorded: set[int] = set()

    def initialize(
        self,
        output_dir: Path,
        field_configs: list[tuple[str, str]],
        dpi: int,
        figsize: tuple[int, int],
        *,
        expected_num_frames: int | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.dpi = dpi
        self.figsize = figsize

        for field_name, colormap in field_configs:
            self.spacetime_data[field_name] = []
            self.field_colormaps[field_name] = colormap

    def save_frame(
        self,
        data: np.ndarray,
        field_name: str,
        frame_index: int,
        simulation_time: float,
        vmin: float | None,
        vmax: float | None,
        colormap: str,
    ) -> None:
        if field_name not in self.spacetime_data:
            self.spacetime_data[field_name] = []

        self.spacetime_data[field_name].append(np.asarray(data).copy())

        if frame_index not in self._times_recorded:
            self.times.append(simulation_time)
            self._times_recorded.add(frame_index)

    def finalize(self) -> dict[str, Any]:
        if not self.spacetime_data or self.output_dir is None:
            return {"format": self.output_format, "error": "No data to save"}

        output_files: dict[str, str] = {}

        for field_name, frames in self.spacetime_data.items():
            if not frames:
                continue

            spacetime = np.array(frames)  # (T, X)
            vmin = float(np.min(spacetime))
            vmax = float(np.max(spacetime))
            colormap = self.field_colormaps.get(field_name, "viridis")

            # Space-time diagram is always written
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            im = ax.imshow(
                spacetime,
                aspect="auto",
                origin="lower",
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
                extent=[
                    0,
                    spacetime.shape[1],
                    0,
                    self.times[-1] if self.times else 1.0,
                ],
            )
            ax.set_xlabel("Space (x)")
            ax.set_ylabel("Time (t)")
            ax.set_title(f"{field_name} - Space-Time Diagram")
            plt.colorbar(im, ax=ax, label=field_name)

            st_filename = f"{field_name}_spacetime.png"
            fig.savefig(self.output_dir / st_filename, bbox_inches="tight")
            plt.close(fig)
            output_files[f"{field_name}_spacetime"] = st_filename

            # Optional animation for 1D fields
            if self.output_format in ("gif", "mp4"):
                filename = f"{field_name}.{self.output_format}"
                out_path = self.output_dir / filename
                if self.output_format == "mp4":
                    writer = imageio.get_writer(
                        out_path,
                        fps=self.fps,
                        codec="libx264",
                        output_params=["-pix_fmt", "yuv420p"],
                    )
                else:
                    writer = imageio.get_writer(
                        out_path,
                        mode="I",
                        duration=1.0 / self.fps,
                        loop=0,
                    )

                for i in range(spacetime.shape[0]):
                    fig, ax = plt.subplots(figsize=(6, 4), dpi=self.dpi)
                    ax.plot(spacetime[i])
                    ax.set_xlabel("Space (x)")
                    ax.set_ylabel(field_name)
                    ax.set_title(
                        f"t = {self.times[i]:.4f}" if i < len(self.times) else ""
                    )
                    fig.canvas.draw()
                    frame_rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
                    writer.append_data(frame_rgb)
                    plt.close(fig)

                writer.close()
                output_files[field_name] = filename

        return {
            "format": self.output_format,
            "type": "1D_spacetime",
            "files": output_files,
        }


class PNGHandler(OutputHandler):
    """Saves PNG frames to per-field subdirectories (2D only)."""

    def __init__(self) -> None:
        self.output_dir: Path | None = None
        self.frames_dir: Path | None = None
        self.field_dirs: dict[str, Path] = {}

    def initialize(
        self,
        output_dir: Path,
        field_configs: list[tuple[str, str]],
        dpi: int,
        figsize: tuple[int, int],
        *,
        expected_num_frames: int | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.frames_dir = output_dir / "frames"

        for field_name, _ in field_configs:
            field_dir = self.frames_dir / field_name
            field_dir.mkdir(parents=True, exist_ok=True)
            self.field_dirs[field_name] = field_dir

    def save_frame(
        self,
        data: np.ndarray,
        field_name: str,
        frame_index: int,
        simulation_time: float,
        vmin: float | None,
        vmax: float | None,
        colormap: str,
    ) -> None:
        if self.frames_dir is None:
            raise RuntimeError("PNGHandler not initialized")

        if field_name not in self.field_dirs:
            field_dir = self.frames_dir / field_name
            field_dir.mkdir(parents=True, exist_ok=True)
            self.field_dirs[field_name] = field_dir

        frame_rgb = render_colormap_rgb(data, vmin=vmin, vmax=vmax, colormap=colormap)
        frame_path = self.field_dirs[field_name] / f"{frame_index:06d}.png"
        imageio.imwrite(frame_path, frame_rgb)

    def finalize(self) -> dict[str, Any]:
        return {
            "format": "png",
            "framesDirectory": "frames/",
            "fields": list(self.field_dirs.keys()),
        }


class MP4Handler(OutputHandler):
    """Streams frames into per-field MP4 files."""

    def __init__(self, fps: int = 30) -> None:
        self.fps = fps
        self.output_dir: Path | None = None
        self._writers: dict[str, Any] = {}

    def initialize(
        self,
        output_dir: Path,
        field_configs: list[tuple[str, str]],
        dpi: int,
        figsize: tuple[int, int],
        *,
        expected_num_frames: int | None = None,
    ) -> None:
        self.output_dir = output_dir
        for field_name, _cmap in field_configs:
            video_path = output_dir / f"{field_name}.mp4"
            self._writers[field_name] = imageio.get_writer(
                video_path,
                fps=self.fps,
                codec="libx264",
                output_params=["-pix_fmt", "yuv420p"],
            )

    def save_frame(
        self,
        data: np.ndarray,
        field_name: str,
        frame_index: int,
        simulation_time: float,
        vmin: float | None,
        vmax: float | None,
        colormap: str,
    ) -> None:
        if field_name not in self._writers:
            raise KeyError(f"Unknown field '{field_name}' for MP4 output")

        frame_rgb = render_colormap_rgb(data, vmin=vmin, vmax=vmax, colormap=colormap)
        self._writers[field_name].append_data(frame_rgb)

    def finalize(self) -> dict[str, Any]:
        video_files: dict[str, str] = {}
        for field_name, writer in self._writers.items():
            try:
                writer.close()
            except Exception:
                pass
            video_files[field_name] = f"{field_name}.mp4"

        return {
            "format": "mp4",
            "fps": self.fps,
            "videos": video_files,
        }


class GIFHandler(OutputHandler):
    """Streams frames into per-field looping GIF files."""

    def __init__(self, fps: int = 30) -> None:
        self.fps = fps
        self.output_dir: Path | None = None
        self._writers: dict[str, Any] = {}

    def initialize(
        self,
        output_dir: Path,
        field_configs: list[tuple[str, str]],
        dpi: int,
        figsize: tuple[int, int],
        *,
        expected_num_frames: int | None = None,
    ) -> None:
        self.output_dir = output_dir
        duration = 1.0 / self.fps
        for field_name, _cmap in field_configs:
            gif_path = output_dir / f"{field_name}.gif"
            self._writers[field_name] = imageio.get_writer(
                gif_path, mode="I", duration=duration, loop=0
            )

    def save_frame(
        self,
        data: np.ndarray,
        field_name: str,
        frame_index: int,
        simulation_time: float,
        vmin: float | None,
        vmax: float | None,
        colormap: str,
    ) -> None:
        if field_name not in self._writers:
            raise KeyError(f"Unknown field '{field_name}' for GIF output")

        frame_rgb = render_colormap_rgb(data, vmin=vmin, vmax=vmax, colormap=colormap)
        self._writers[field_name].append_data(frame_rgb)

    def finalize(self) -> dict[str, Any]:
        gif_files: dict[str, str] = {}
        for field_name, writer in self._writers.items():
            try:
                writer.close()
            except Exception:
                pass
            gif_files[field_name] = f"{field_name}.gif"

        return {
            "format": "gif",
            "fps": self.fps,
            "gifs": gif_files,
        }


class NumpyHandler(OutputHandler):
    """Writes all field data into `trajectory.npy` and `times.npy`.

    When `expected_num_frames` is supplied, this uses `numpy.lib.format.open_memmap`
    to stream writes without buffering all frames in RAM.
    """

    def __init__(self) -> None:
        self.output_dir: Path | None = None
        self.field_order: list[str] = []
        self._field_index: dict[str, int] = {}

        self._expected_num_frames: int | None = None
        self._trajectory: np.memmap | None = None
        self._times: np.memmap | None = None
        self._trajectory_path: Path | None = None
        self._times_path: Path | None = None
        self._spatial_shape: tuple[int, ...] | None = None
        self._written_frames: set[int] = set()

        # Fallback buffers (when expected frame count is unknown)
        self._buffer: dict[str, list[np.ndarray]] = {}
        self._buffer_times: list[float] = []
        self._times_recorded: set[int] = set()

    def initialize(
        self,
        output_dir: Path,
        field_configs: list[tuple[str, str]],
        dpi: int,
        figsize: tuple[int, int],
        *,
        expected_num_frames: int | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.field_order = [name for name, _ in field_configs]
        self._field_index = {name: i for i, name in enumerate(self.field_order)}
        self._expected_num_frames = expected_num_frames

        # Only allocate files lazily once we know the spatial shape from data.
        self._buffer = {name: [] for name in self.field_order}

    def _ensure_memmaps(self, spatial_shape: tuple[int, ...]) -> None:
        if self.output_dir is None:
            raise RuntimeError("NumpyHandler not initialized")
        if self._trajectory is not None:
            return
        if self._expected_num_frames is None:
            return  # buffering mode

        T = int(self._expected_num_frames)
        F = len(self.field_order)
        self._spatial_shape = spatial_shape

        self._trajectory_path = self.output_dir / "trajectory.npy"
        self._times_path = self.output_dir / "times.npy"

        self._trajectory = open_memmap(
            self._trajectory_path, mode="w+", dtype=np.float64, shape=(T, *spatial_shape, F)
        )
        self._times = open_memmap(
            self._times_path, mode="w+", dtype=np.float64, shape=(T,)
        )

    def save_frame(
        self,
        data: np.ndarray,
        field_name: str,
        frame_index: int,
        simulation_time: float,
        vmin: float | None,
        vmax: float | None,
        colormap: str,
    ) -> None:
        data = np.asarray(data)

        if field_name not in self._field_index:
            self._field_index[field_name] = len(self.field_order)
            self.field_order.append(field_name)
            self._buffer.setdefault(field_name, [])

        if self._expected_num_frames is not None:
            # Streaming mode
            self._ensure_memmaps(data.shape)
            if self._trajectory is None or self._times is None:
                raise RuntimeError("Failed to initialize memmaps for numpy output")

            f_idx = self._field_index[field_name]
            if f_idx >= self._trajectory.shape[-1]:
                raise ValueError(
                    "Unexpected field encountered after allocation. "
                    "Pass full field_configs to OutputManager."
                )
            if frame_index >= self._trajectory.shape[0]:
                raise ValueError(
                    f"Frame index {frame_index} exceeds expected_num_frames={self._trajectory.shape[0]}"
                )

            self._trajectory[frame_index, ..., f_idx] = data
            if frame_index not in self._written_frames:
                self._times[frame_index] = simulation_time
                self._written_frames.add(frame_index)
            return

        # Buffering mode (used by unit tests and legacy call-sites)
        self._buffer[field_name].append(data.copy())
        if frame_index not in self._times_recorded:
            self._buffer_times.append(simulation_time)
            self._times_recorded.add(frame_index)

    def finalize(self) -> dict[str, Any]:
        if self.output_dir is None:
            return {"format": "numpy", "error": "No output directory"}

        # Streaming mode: files are already written
        if self._trajectory is not None and self._times is not None:
            # Ensure data is flushed to disk
            self._trajectory.flush()
            self._times.flush()

            # Compute per-field ranges without loading everything into memory
            field_ranges: dict[str, dict[str, float]] = {}
            for f_idx, field_name in enumerate(self.field_order):
                field_data = self._trajectory[..., f_idx]
                field_ranges[field_name] = {
                    "min": float(np.min(field_data)),
                    "max": float(np.max(field_data)),
                }

            ndim = len(self._spatial_shape or ())
            shape_desc = "(T, X, F)" if ndim == 1 else "(T, H, W, F)" if ndim == 2 else "(T, D, H, W, F)"

            return {
                "format": "numpy",
                "trajectoryFile": "trajectory.npy",
                "timesFile": "times.npy",
                "shape": list(self._trajectory.shape),
                "shapeDescription": shape_desc,
                "ndim": ndim,
                "fieldOrder": self.field_order,
                "fieldRanges": field_ranges,
                "dtype": str(self._trajectory.dtype),
            }

        # Buffering mode: combine and save now
        if not self.field_order or not self._buffer[self.field_order[0]]:
            return {"format": "numpy", "error": "No data to save"}

        T = len(self._buffer_times)
        first_field_data = self._buffer[self.field_order[0]]
        spatial_shape = first_field_data[0].shape
        F = len(self.field_order)
        ndim = len(spatial_shape)

        array = np.zeros((T, *spatial_shape, F), dtype=np.float64)
        for f_idx, field_name in enumerate(self.field_order):
            frames = self._buffer[field_name]
            for t_idx, arr in enumerate(frames):
                array[(t_idx, ..., f_idx)] = arr

        np.save(self.output_dir / "trajectory.npy", array)
        np.save(self.output_dir / "times.npy", np.array(self._buffer_times))

        field_ranges: dict[str, dict[str, float]] = {}
        for f_idx, field_name in enumerate(self.field_order):
            field_data = array[..., f_idx]
            field_ranges[field_name] = {
                "min": float(np.min(field_data)),
                "max": float(np.max(field_data)),
            }

        shape_desc = "(T, X, F)" if ndim == 1 else "(T, H, W, F)" if ndim == 2 else "(T, D, H, W, F)"
        return {
            "format": "numpy",
            "trajectoryFile": "trajectory.npy",
            "timesFile": "times.npy",
            "shape": list(array.shape),
            "shapeDescription": shape_desc,
            "ndim": ndim,
            "fieldOrder": self.field_order,
            "fieldRanges": field_ranges,
            "dtype": str(array.dtype),
        }


class H5Handler(OutputHandler):
    """Streams all field data into a single `trajectory.h5` file.

    Dataset layout matches the numpy format: (T, *spatial_shape, F)
    """

    def __init__(self, compression: bool = True) -> None:
        self.compression = compression
        self.output_dir: Path | None = None
        self.field_order: list[str] = []
        self._field_index: dict[str, int] = {}
        self._expected_num_frames: int | None = None

        self._h5: Any = None
        self._traj: Any = None
        self._times: Any = None
        self._spatial_shape: tuple[int, ...] | None = None
        self._written_frames: set[int] = set()

    def initialize(
        self,
        output_dir: Path,
        field_configs: list[tuple[str, str]],
        dpi: int,
        figsize: tuple[int, int],
        *,
        expected_num_frames: int | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.field_order = [name for name, _ in field_configs]
        self._field_index = {name: i for i, name in enumerate(self.field_order)}
        self._expected_num_frames = expected_num_frames

        path = output_dir / "trajectory.h5"
        self._h5 = h5py.File(path, "w")
        self._h5.attrs["fieldOrder"] = json.dumps(self.field_order)

    def _ensure_datasets(self, spatial_shape: tuple[int, ...]) -> None:
        if self._traj is not None:
            return
        if self._h5 is None:
            raise RuntimeError("H5Handler not initialized")

        self._spatial_shape = spatial_shape
        F = len(self.field_order)
        chunks = (1, *spatial_shape, F)

        compression = "gzip" if self.compression else None
        if self._expected_num_frames is not None:
            T = int(self._expected_num_frames)
            self._traj = self._h5.create_dataset(
                "trajectory",
                shape=(T, *spatial_shape, F),
                dtype=np.float64,
                chunks=chunks,
                compression=compression,
            )
            self._times = self._h5.create_dataset(
                "times",
                shape=(T,),
                dtype=np.float64,
                chunks=(min(1024, T),),
                compression=compression,
            )
        else:
            self._traj = self._h5.create_dataset(
                "trajectory",
                shape=(0, *spatial_shape, F),
                maxshape=(None, *spatial_shape, F),
                dtype=np.float64,
                chunks=chunks,
                compression=compression,
            )
            self._times = self._h5.create_dataset(
                "times",
                shape=(0,),
                maxshape=(None,),
                dtype=np.float64,
                chunks=(1024,),
                compression=compression,
            )

    def save_frame(
        self,
        data: np.ndarray,
        field_name: str,
        frame_index: int,
        simulation_time: float,
        vmin: float | None,
        vmax: float | None,
        colormap: str,
    ) -> None:
        data = np.asarray(data)
        if field_name not in self._field_index:
            # keep file stable: unexpected fields are a hard error
            raise KeyError(
                f"Unknown field '{field_name}' for h5 output. "
                "Provide complete field_configs to OutputManager."
            )
        self._ensure_datasets(data.shape)
        assert self._traj is not None and self._times is not None

        if self._expected_num_frames is None and frame_index >= self._traj.shape[0]:
            new_len = frame_index + 1
            self._traj.resize((new_len, *self._traj.shape[1:]))
            self._times.resize((new_len,))

        f_idx = self._field_index[field_name]
        self._traj[frame_index, ..., f_idx] = data
        if frame_index not in self._written_frames:
            self._times[frame_index] = simulation_time
            self._written_frames.add(frame_index)

    def finalize(self) -> dict[str, Any]:
        ndim = len(self._spatial_shape or ())
        # Capture dataset shape before closing the file (h5py objects become invalid).
        shape = list(self._traj.shape) if self._traj is not None else None
        shape_desc = "(T, X, F)" if ndim == 1 else "(T, H, W, F)" if ndim == 2 else "(T, D, H, W, F)"

        if self._h5 is not None:
            try:
                self._h5.flush()
                self._h5.close()
            except Exception:
                pass

        return {
            "format": "h5",
            "trajectoryFile": "trajectory.h5",
            "dataset": "trajectory",
            "timesDataset": "times",
            "shape": shape,
            "shapeDescription": shape_desc,
            "ndim": ndim,
            "fieldOrder": self.field_order,
        }


def create_output_handler(output_format: str, fps: int = 30, ndim: int = 2) -> OutputHandler:
    """Factory for output handlers."""
    if ndim == 1:
        if output_format in ("png", "mp4", "gif"):
            return Output1DHandler(output_format=output_format, fps=fps)
        if output_format in ("numpy", "h5"):
            return NumpyHandler() if output_format == "numpy" else H5Handler()
        raise ValueError(f"Unknown output format: {output_format}")

    if ndim == 2:
        if output_format == "png":
            return PNGHandler()
        if output_format == "mp4":
            return MP4Handler(fps=fps)
        if output_format == "gif":
            return GIFHandler(fps=fps)
        if output_format == "numpy":
            return NumpyHandler()
        if output_format == "h5":
            return H5Handler()
        raise ValueError(f"Unknown output format: {output_format}")

    if ndim == 3:
        # 3D visualization deferred; allow trajectory formats only
        if output_format == "numpy":
            return NumpyHandler()
        if output_format == "h5":
            return H5Handler()
        raise ValueError(
            f"3D visualization not yet supported. Use format 'numpy' or 'h5' instead of '{output_format}'"
        )

    raise ValueError(f"Unsupported dimensionality: {ndim}D. Must be 1, 2, or 3.")

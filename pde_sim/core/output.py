"""Output management for simulation frames and metadata."""

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless systems
import matplotlib.pyplot as plt
import numpy as np
from pde import FieldCollection, ScalarField

from pde_sim.descriptions import get_description


class OutputHandler(ABC):
    """Abstract base class for output format handlers."""

    @abstractmethod
    def initialize(
        self,
        output_dir: Path,
        field_configs: list[tuple[str, str]],
        dpi: int,
        figsize: tuple[int, int],
    ) -> None:
        """Set up output directories/files.

        Args:
            output_dir: Base output directory.
            field_configs: List of (field_name, colormap) tuples.
            dpi: DPI for rendered images.
            figsize: Figure size in inches.
        """
        pass

    @abstractmethod
    def save_frame(
        self,
        data: np.ndarray,
        field_name: str,
        frame_index: int,
        simulation_time: float,
        vmin: float,
        vmax: float,
        colormap: str,
    ) -> None:
        """Save a single frame for one field.

        Args:
            data: 2D numpy array of field values.
            field_name: Name of the field.
            frame_index: Index of this frame.
            simulation_time: Simulation time at this frame.
            vmin: Minimum value for colormap.
            vmax: Maximum value for colormap.
            colormap: Matplotlib colormap name.
        """
        pass

    @abstractmethod
    def finalize(self) -> dict[str, Any]:
        """Complete output (e.g., encode video) and return format-specific metadata.

        Returns:
            Dictionary with format-specific metadata to include in metadata.json.
        """
        pass


class PNGHandler(OutputHandler):
    """Saves PNG frames to per-field subdirectories."""

    def __init__(self) -> None:
        self.output_dir: Path | None = None
        self.frames_dir: Path | None = None
        self.field_dirs: dict[str, Path] = {}
        self.dpi: int = 128
        self.figsize: tuple[int, int] = (8, 8)

    def initialize(
        self,
        output_dir: Path,
        field_configs: list[tuple[str, str]],
        dpi: int,
        figsize: tuple[int, int],
    ) -> None:
        self.output_dir = output_dir
        self.frames_dir = output_dir / "frames"
        self.dpi = dpi
        self.figsize = figsize

        # Create per-field subdirectories
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
        vmin: float,
        vmax: float,
        colormap: str,
    ) -> None:
        if field_name not in self.field_dirs:
            # Create directory for unexpected field
            field_dir = self.frames_dir / field_name
            field_dir.mkdir(parents=True, exist_ok=True)
            self.field_dirs[field_name] = field_dir

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.imshow(
            data.T,  # Transpose for correct orientation
            origin="lower",
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        ax.axis("off")

        # Save to field subdirectory: frames/{field}/{frame:06d}.png
        frame_path = self.field_dirs[field_name] / f"{frame_index:06d}.png"
        fig.savefig(frame_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def finalize(self) -> dict[str, Any]:
        return {
            "format": "png",
            "framesDirectory": "frames/",
            "fields": list(self.field_dirs.keys()),
        }


class MP4Handler(OutputHandler):
    """Accumulates frames and encodes MP4 videos per field."""

    def __init__(self, fps: int = 30) -> None:
        self.fps = fps
        self.output_dir: Path | None = None
        self.frame_buffers: dict[str, list[np.ndarray]] = {}
        self.dpi: int = 128
        self.figsize: tuple[int, int] = (8, 8)
        self.field_colormaps: dict[str, str] = {}

    def initialize(
        self,
        output_dir: Path,
        field_configs: list[tuple[str, str]],
        dpi: int,
        figsize: tuple[int, int],
    ) -> None:
        self.output_dir = output_dir
        self.dpi = dpi
        self.figsize = figsize

        for field_name, colormap in field_configs:
            self.frame_buffers[field_name] = []
            self.field_colormaps[field_name] = colormap

    def save_frame(
        self,
        data: np.ndarray,
        field_name: str,
        frame_index: int,
        simulation_time: float,
        vmin: float,
        vmax: float,
        colormap: str,
    ) -> None:
        if field_name not in self.frame_buffers:
            self.frame_buffers[field_name] = []

        # Render frame to numpy array (RGB)
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.imshow(
            data.T,
            origin="lower",
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        ax.axis("off")

        # Draw canvas and convert to RGB array
        fig.canvas.draw()
        # Use buffer_rgba() and discard alpha channel (modern matplotlib API)
        frame_data = np.asarray(fig.canvas.buffer_rgba())
        frame_data = frame_data[:, :, :3]  # Keep only RGB, discard alpha
        plt.close(fig)

        self.frame_buffers[field_name].append(frame_data)

    def finalize(self) -> dict[str, Any]:
        import imageio

        video_files: dict[str, str] = {}
        for field_name, frames in self.frame_buffers.items():
            if frames:
                video_path = self.output_dir / f"{field_name}.mp4"
                imageio.mimwrite(
                    video_path,
                    frames,
                    fps=self.fps,
                    codec="libx264",
                    output_params=["-pix_fmt", "yuv420p"],
                )
                video_files[field_name] = f"{field_name}.mp4"

        return {
            "format": "mp4",
            "fps": self.fps,
            "videos": video_files,
        }


class NumpyHandler(OutputHandler):
    """Collects all field data into a single (T, H, W, F) array."""

    def __init__(self) -> None:
        self.output_dir: Path | None = None
        self.trajectory_data: dict[str, list[np.ndarray]] = {}
        self.times: list[float] = []
        self.field_order: list[str] = []
        self._times_recorded: set[int] = set()  # Track which frame indices have times

    def initialize(
        self,
        output_dir: Path,
        field_configs: list[tuple[str, str]],
        dpi: int,
        figsize: tuple[int, int],
    ) -> None:
        self.output_dir = output_dir
        self.field_order = [name for name, _ in field_configs]
        for field_name in self.field_order:
            self.trajectory_data[field_name] = []

    def save_frame(
        self,
        data: np.ndarray,
        field_name: str,
        frame_index: int,
        simulation_time: float,
        vmin: float,
        vmax: float,
        colormap: str,
    ) -> None:
        if field_name not in self.trajectory_data:
            self.trajectory_data[field_name] = []
            self.field_order.append(field_name)

        self.trajectory_data[field_name].append(data.copy())

        # Track time only once per frame index
        if frame_index not in self._times_recorded:
            self.times.append(simulation_time)
            self._times_recorded.add(frame_index)

    def finalize(self) -> dict[str, Any]:
        if not self.field_order or not self.trajectory_data[self.field_order[0]]:
            return {"format": "numpy", "error": "No data to save"}

        T = len(self.times)
        first_field_data = self.trajectory_data[self.field_order[0]]
        H, W = first_field_data[0].shape
        F = len(self.field_order)

        # Create combined array with shape (T, H, W, F)
        array = np.zeros((T, H, W, F), dtype=np.float64)
        for f_idx, field_name in enumerate(self.field_order):
            field_frames = self.trajectory_data[field_name]
            for t_idx, data in enumerate(field_frames):
                array[t_idx, :, :, f_idx] = data

        # Save trajectory and times
        np.save(self.output_dir / "trajectory.npy", array)
        np.save(self.output_dir / "times.npy", np.array(self.times))

        # Compute field ranges for metadata
        field_ranges = {}
        for f_idx, field_name in enumerate(self.field_order):
            field_data = array[:, :, :, f_idx]
            field_ranges[field_name] = {
                "min": float(np.min(field_data)),
                "max": float(np.max(field_data)),
            }

        return {
            "format": "numpy",
            "trajectoryFile": "trajectory.npy",
            "timesFile": "times.npy",
            "shape": list(array.shape),
            "fieldOrder": self.field_order,
            "fieldRanges": field_ranges,
            "dtype": str(array.dtype),
        }


def create_output_handler(output_format: str, fps: int = 30) -> OutputHandler:
    """Factory function to create appropriate output handler.

    Args:
        output_format: Output format ("png", "mp4", or "numpy").
        fps: Frame rate for MP4 output.

    Returns:
        Appropriate OutputHandler instance.

    Raises:
        ValueError: If format is not recognized.
    """
    handlers = {
        "png": PNGHandler,
        "mp4": lambda: MP4Handler(fps=fps),
        "numpy": NumpyHandler,
    }

    if output_format not in handlers:
        raise ValueError(
            f"Unknown output format: {output_format}. "
            f"Valid formats: {list(handlers.keys())}"
        )

    handler_class = handlers[output_format]
    if callable(handler_class) and not isinstance(handler_class, type):
        return handler_class()
    return handler_class()


class OutputManager:
    """Manages simulation output (frames and metadata).

    Uses the strategy pattern to support multiple output formats:
    - png: PNG images in per-field subdirectories
    - mp4: MP4 videos per field
    - numpy: Single numpy array with shape (T, H, W, F)
    """

    def __init__(
        self,
        base_path: Path | str,
        folder_name: str,
        colormap: str = "turbo",
        field_configs: list[tuple[str, str]] | None = None,
        dpi: int = 128,
        figsize: tuple[int, int] = (8, 8),
        output_format: str = "png",
        fps: int = 30,
    ):
        """Initialize the output manager.

        Args:
            base_path: Base directory for output.
            folder_name: Path for the output folder (e.g., "gray-scott/001").
            colormap: Default matplotlib colormap name.
            field_configs: List of (field_name, colormap) tuples for output.
            dpi: DPI for saved images.
            figsize: Figure size in inches.
            output_format: Output format ("png", "mp4", or "numpy").
            fps: Frame rate for MP4 output.
        """
        self.base_path = Path(base_path)
        self.folder_name = folder_name
        self.colormap = colormap
        self.dpi = dpi
        self.figsize = figsize
        self.output_format = output_format
        self.fps = fps

        # Field configurations: [(field_name, colormap), ...]
        self.field_configs = field_configs or []

        # Per-field range tracking: field_name -> (vmin, vmax)
        self.field_ranges: dict[str, tuple[float, float]] = {}

        # Per-field colormap lookup
        self.field_colormaps: dict[str, str] = {}
        for field_name, cmap in self.field_configs:
            self.field_colormaps[field_name] = cmap

        # Create output directory
        self.output_dir = self.base_path / self.folder_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create appropriate handler
        self.handler = create_output_handler(output_format, fps=fps)
        self.handler.initialize(self.output_dir, self.field_configs, dpi, figsize)

        # Track saved frames
        self.saved_frames: list[dict[str, Any]] = []

    def compute_range_for_field(
        self,
        states: list[ScalarField | FieldCollection],
        field_name: str,
    ) -> tuple[float, float]:
        """Compute global min/max for a specific field across all states.

        Args:
            states: List of field states.
            field_name: Name of the field to compute range for.

        Returns:
            Tuple of (vmin, vmax) for this field.
        """
        global_min = float("inf")
        global_max = float("-inf")

        for state in states:
            data = self._extract_field_data(state, field_name)
            global_min = min(global_min, float(np.min(data)))
            global_max = max(global_max, float(np.max(data)))

        self.field_ranges[field_name] = (global_min, global_max)
        return (global_min, global_max)

    def _extract_field_data(
        self,
        state: ScalarField | FieldCollection,
        field_name: str,
    ) -> np.ndarray:
        """Extract data for a specific field by name.

        Args:
            state: The field state.
            field_name: Name of the field to extract.

        Returns:
            2D numpy array of field values.
        """
        if isinstance(state, FieldCollection):
            data = self._get_field_by_name(state, field_name)
            if data is None:
                # Fallback to first field
                data = state[0].data
        else:
            data = state.data

        # Handle complex-valued fields
        if np.iscomplexobj(data):
            data = np.abs(data)

        return data

    def _get_field_by_name(
        self, state: FieldCollection, field_name: str
    ) -> np.ndarray | None:
        """Get field data from a FieldCollection by name.

        Args:
            state: The FieldCollection to search.
            field_name: Name (label) of the field to find.

        Returns:
            Field data array or None if not found.
        """
        for field in state:
            if hasattr(field, "label") and field.label == field_name:
                return field.data
        # Try index
        try:
            idx = int(field_name)
            return state[idx].data
        except (ValueError, IndexError):
            return None

    def save_frame(
        self,
        state: ScalarField | FieldCollection,
        field_name: str,
        frame_index: int,
        simulation_time: float,
    ) -> None:
        """Save a single frame for a specific field.

        Args:
            state: The field state.
            field_name: Name of the field to save.
            frame_index: Index of this frame.
            simulation_time: Simulation time at this frame.
        """
        data = self._extract_field_data(state, field_name)

        # Get field-specific colormap and range
        colormap = self.field_colormaps.get(field_name, self.colormap)
        vmin, vmax = self.field_ranges.get(field_name, (None, None))

        self.handler.save_frame(
            data, field_name, frame_index, simulation_time, vmin, vmax, colormap
        )

    def save_all_fields(
        self,
        state: ScalarField | FieldCollection,
        frame_index: int,
        simulation_time: float,
    ) -> None:
        """Save frames for all configured fields.

        Args:
            state: The field state.
            frame_index: Index of this frame.
            simulation_time: Simulation time.
        """
        for field_name, _colormap in self.field_configs:
            self.save_frame(state, field_name, frame_index, simulation_time)

        # Track frame (once per timestep, not per field)
        self.saved_frames.append({
            "frameIndex": frame_index,
            "simulationTime": simulation_time,
        })

    def finalize(self) -> dict[str, Any]:
        """Finalize output and return format-specific metadata.

        Returns:
            Dictionary with format-specific metadata.
        """
        return self.handler.finalize()

    def save_metadata(self, metadata: dict[str, Any]) -> Path:
        """Save metadata JSON.

        Args:
            metadata: Metadata dictionary to save.

        Returns:
            Path to the saved metadata file.
        """
        # Add per-field visualization info
        if "visualization" not in metadata:
            metadata["visualization"] = {}

        metadata["visualization"]["fields"] = {}
        for field_name, colormap in self.field_configs:
            vmin, vmax = self.field_ranges.get(field_name, (0.0, 1.0))
            metadata["visualization"]["fields"][field_name] = {
                "colormap": colormap,
                "colorbarMin": vmin,
                "colorbarMax": vmax,
            }

        # Add format-specific output info
        metadata["output"] = self.finalize()

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata_path

    def get_frame_annotations(self) -> list[dict[str, Any]]:
        """Get frame annotations for metadata.

        Returns:
            List of frame annotation dictionaries.
        """
        return self.saved_frames.copy()


def _bc_to_metadata(bc: Any) -> dict[str, Any]:
    """Convert BoundaryConfig to metadata dictionary.

    Args:
        bc: BoundaryConfig object

    Returns:
        Dictionary with boundary condition information for metadata.
    """
    result: dict[str, Any] = {
        "x-": bc.x_minus,
        "x+": bc.x_plus,
        "y-": bc.y_minus,
        "y+": bc.y_plus,
    }

    if bc.fields:
        result["fields"] = bc.fields

    return result


def _create_visualization_metadata(config: Any, preset_metadata: Any) -> dict[str, Any]:
    """Create visualization metadata section.

    Args:
        config: SimulationConfig object.
        preset_metadata: PDEMetadata from the preset.

    Returns:
        Visualization metadata dictionary.
    """
    from .config import COLORMAP_CYCLE

    # All fields are always output with auto-assigned colormaps
    field_names = preset_metadata.field_names
    return {
        "whatToPlot": field_names,
        "colormaps": {
            name: COLORMAP_CYCLE[i % len(COLORMAP_CYCLE)]
            for i, name in enumerate(field_names)
        },
    }


def create_metadata(
    sim_id: str,
    preset_name: str,
    preset_metadata: Any,
    config: Any,
    total_time: float,
    frame_annotations: list[dict[str, Any]],
    solver_diagnostics: dict[str, Any] | None = None,
    wall_clock_duration: float | None = None,
) -> dict[str, Any]:
    """Create a complete metadata dictionary.

    Args:
        sim_id: Unique simulation ID.
        preset_name: Name of the PDE preset.
        preset_metadata: PDEMetadata from the preset.
        config: SimulationConfig object.
        total_time: Total simulation time.
        frame_annotations: List of per-frame annotations.
        solver_diagnostics: Optional solver diagnostics from py-pde.
        wall_clock_duration: Wall clock time in seconds for the simulation.

    Returns:
        Complete metadata dictionary.
    """
    solver_diagnostics = solver_diagnostics or {}

    # Convert dt_statistics numpy values to Python floats for JSON serialization
    dt_stats = solver_diagnostics.get("dt_statistics")
    if dt_stats:
        dt_stats = {k: float(v) if hasattr(v, 'item') else v for k, v in dt_stats.items()}

    # Load detailed description from markdown file
    description = get_description(preset_name)

    return {
        "id": sim_id,
        "preset": preset_name,
        "description": description,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generatorVersion": "1.0.0",
        "equations": preset_metadata.equations,
        "boundaryConditions": _bc_to_metadata(config.bc),
        "initialConditions": config.init.type,
        "parameters": {
            "kinetic": config.parameters,
            "dt": config.dt,
            "spatialStep": config.domain_size / config.resolution,
            "domainScale": str(config.domain_size),
            "timesteppingScheme": config.solver.title(),
            "numSpecies": preset_metadata.num_fields,
            "backend": config.backend,
            "adaptive": config.adaptive,
            "tolerance": config.tolerance if config.adaptive else None,
        },
        "simulation": {
            "totalFrames": len(frame_annotations),
            "numFrames": config.output.num_frames,
            "totalTime": total_time,
            "wallClockDuration": wall_clock_duration,
            "resolution": [config.resolution, config.resolution],
            "dtStatistics": dt_stats,
        },
        "visualization": _create_visualization_metadata(config, preset_metadata),
        "interventions": [],
        "frameAnnotations": frame_annotations,
    }

"""Output management for simulation frames and metadata."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pde import FieldCollection, ScalarField

from pde_sim.descriptions import get_description


class OutputManager:
    """Manages simulation output (frames and metadata).

    Handles saving PNG frames and JSON metadata for each simulation.
    """

    def __init__(
        self,
        base_path: Path | str,
        folder_name: str,
        colormap: str = "turbo",
        field_configs: list[tuple[str, str]] | None = None,
        dpi: int = 128,
        figsize: tuple[int, int] = (8, 8),
        save_array: bool = False,
    ):
        """Initialize the output manager.

        Args:
            base_path: Base directory for output.
            folder_name: Path for the output folder (e.g., "gray-scott/001").
            colormap: Default matplotlib colormap name.
            field_configs: List of (field_name, colormap) tuples for output.
            dpi: DPI for saved images.
            figsize: Figure size in inches.
            save_array: Whether to save trajectory as numpy array.
        """
        self.base_path = Path(base_path)
        self.folder_name = folder_name
        self.colormap = colormap
        self.dpi = dpi
        self.figsize = figsize
        self.save_array = save_array

        # Field configurations: [(field_name, colormap), ...]
        self.field_configs = field_configs or []

        # Per-field range tracking: field_name -> (vmin, vmax)
        self.field_ranges: dict[str, tuple[float, float]] = {}

        # Per-field colormap lookup
        self.field_colormaps: dict[str, str] = {}
        for field_name, cmap in self.field_configs:
            self.field_colormaps[field_name] = cmap

        # Create output directories
        self.output_dir = self.base_path / self.folder_name
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

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
    ) -> Path:
        """Save a single frame for a specific field.

        Args:
            state: The field state.
            field_name: Name of the field to save.
            frame_index: Index of this frame.
            simulation_time: Simulation time at this frame.

        Returns:
            Path to the saved PNG file.
        """
        data = self._extract_field_data(state, field_name)

        # Get field-specific colormap and range
        colormap = self.field_colormaps.get(field_name, self.colormap)
        vmin, vmax = self.field_ranges.get(field_name, (None, None))

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

        # Generate filename with field prefix: {field}_{frame:06d}.png
        frame_path = self.frames_dir / f"{field_name}_{frame_index:06d}.png"

        fig.savefig(frame_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        return frame_path

    def save_all_fields(
        self,
        state: ScalarField | FieldCollection,
        frame_index: int,
        simulation_time: float,
    ) -> list[Path]:
        """Save frames for all configured fields.

        Args:
            state: The field state.
            frame_index: Index of this frame.
            simulation_time: Simulation time.

        Returns:
            List of paths to saved PNG files.
        """
        paths = []
        for field_name, _colormap in self.field_configs:
            path = self.save_frame(state, field_name, frame_index, simulation_time)
            paths.append(path)

        # Track frame (once per timestep, not per field)
        self.saved_frames.append({
            "frameIndex": frame_index,
            "simulationTime": simulation_time,
        })

        return paths

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

    def save_trajectory_array(
        self,
        states: list[ScalarField | FieldCollection],
        times: list[float],
        field_name: str,
    ) -> Path:
        """Save the full trajectory for a field as a numpy array.

        Saves a .npz file containing the trajectory with shape:
        (num_frames, resolution_x, resolution_y)

        Also saves a times array with shape (num_frames,).

        Args:
            states: List of field states from the simulation.
            times: List of simulation times corresponding to each state.
            field_name: Name of the field to save.

        Returns:
            Path to the saved .npz file.
        """
        # Extract field data for each state
        trajectory_data = np.stack([
            self._extract_field_data(state, field_name) for state in states
        ])

        times_array = np.array(times)

        # Save as compressed npz with both trajectory and times
        array_path = self.output_dir / f"trajectory_{field_name}.npz"
        np.savez_compressed(array_path, trajectory=trajectory_data, times=times_array)

        return array_path


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
    field_configs = config.output.get_field_configs()
    if field_configs:
        return {
            "whatToPlot": [fc[0] for fc in field_configs],
            "colormaps": {fc[0]: fc[1] for fc in field_configs},
        }
    else:
        # Default: all fields with default colormap
        return {
            "whatToPlot": preset_metadata.field_names,
            "colormaps": {name: config.output.colormap for name in preset_metadata.field_names},
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

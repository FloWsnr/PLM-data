"""Output management for simulation frames and metadata."""

import json
import re
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
        field_to_plot: str | None = None,
        dpi: int = 128,
        figsize: tuple[int, int] = (8, 8),
        save_array: bool = False,
        show_vectors: bool = False,
        vector_density: int = 16,
    ):
        """Initialize the output manager.

        Args:
            base_path: Base directory for output.
            folder_name: Path for the output folder (e.g., "gray-scott/2024-01-15_143052").
            colormap: Matplotlib colormap name.
            field_to_plot: Name of field to render (for multi-field systems).
                Can be a single field name (e.g., "u") or magnitude syntax (e.g., "mag(X,Y)").
            dpi: DPI for saved images.
            figsize: Figure size in inches.
            save_array: Whether to save trajectory as numpy array.
            show_vectors: Whether to overlay vector arrows (only works with mag()).
            vector_density: Number of arrows per axis (e.g., 16 = 16x16 grid).
        """
        self.base_path = Path(base_path)
        self.folder_name = folder_name
        self.colormap = colormap
        self.field_to_plot = field_to_plot
        self.dpi = dpi
        self.figsize = figsize
        self.save_array = save_array
        self.show_vectors = show_vectors
        self.vector_density = vector_density

        # Parse field_to_plot for magnitude syntax
        self.plot_mode, self.plot_fields = self._parse_field_to_plot(field_to_plot)

        # Create output directories
        self.output_dir = self.base_path / self.folder_name
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # Track min/max values for consistent colormapping
        self.vmin: float | None = None
        self.vmax: float | None = None

        # Track saved frames
        self.saved_frames: list[dict[str, Any]] = []

    def _parse_field_to_plot(
        self, field_to_plot: str | None
    ) -> tuple[str, list[str]]:
        """Parse field_to_plot string to determine plot mode.

        Args:
            field_to_plot: Field specification string.

        Returns:
            Tuple of (plot_mode, field_names) where:
            - plot_mode is "single" or "magnitude"
            - field_names is list of field names to extract
        """
        if field_to_plot is None:
            return ("single", [])

        # Check for magnitude syntax: mag(field1, field2)
        mag_pattern = r"^mag\((\w+),\s*(\w+)\)$"
        match = re.match(mag_pattern, field_to_plot)
        if match:
            return ("magnitude", [match.group(1), match.group(2)])

        # Single field mode
        return ("single", [field_to_plot])

    def compute_range(
        self, states: list[ScalarField | FieldCollection]
    ) -> tuple[float, float]:
        """Pre-compute global min/max range across all states.

        This should be called before saving frames to ensure consistent
        colorscale normalization across the entire trajectory.

        Args:
            states: List of field states to compute range from.

        Returns:
            Tuple of (vmin, vmax) values.
        """
        global_min = float("inf")
        global_max = float("-inf")

        for state in states:
            data = self._extract_field_data(state)
            global_min = min(global_min, float(np.min(data)))
            global_max = max(global_max, float(np.max(data)))

        self.vmin = global_min
        self.vmax = global_max

        return (global_min, global_max)

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

    def _extract_field_data(
        self, state: ScalarField | FieldCollection
    ) -> np.ndarray:
        """Extract the numpy array from a field state.

        Args:
            state: The field state to extract data from.

        Returns:
            2D numpy array of real field values.
        """
        if isinstance(state, FieldCollection):
            if self.plot_mode == "magnitude" and len(self.plot_fields) == 2:
                # Magnitude of two fields: sqrt(f1^2 + f2^2)
                f1 = self._get_field_by_name(state, self.plot_fields[0])
                f2 = self._get_field_by_name(state, self.plot_fields[1])
                if f1 is not None and f2 is not None:
                    return np.sqrt(f1**2 + f2**2)
                # Fallback to first field if magnitude fields not found
                data = state[0].data
            elif self.plot_fields:
                # Single field mode with explicit field name
                data = self._get_field_by_name(state, self.plot_fields[0])
                if data is None:
                    data = state[0].data
            else:
                # Default to first field
                data = state[0].data
        else:
            data = state.data

        # Handle complex-valued fields (e.g., Schrodinger equation)
        # Plot the absolute value (magnitude) for visualization
        if np.iscomplexobj(data):
            data = np.abs(data)

        return data

    def _extract_vector_components(
        self, state: ScalarField | FieldCollection
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Extract vector components for quiver plotting.

        Only works when plot_mode is "magnitude".

        Args:
            state: The field state to extract from.

        Returns:
            Tuple of (u_component, v_component) arrays or None if not applicable.
        """
        if self.plot_mode != "magnitude" or len(self.plot_fields) != 2:
            return None

        if not isinstance(state, FieldCollection):
            return None

        f1 = self._get_field_by_name(state, self.plot_fields[0])
        f2 = self._get_field_by_name(state, self.plot_fields[1])

        if f1 is None or f2 is None:
            return None

        return (f1, f2)

    def save_frame(
        self,
        state: ScalarField | FieldCollection,
        frame_index: int,
        simulation_time: float,
    ) -> Path:
        """Save a single frame as PNG.

        Args:
            state: The field state to save.
            frame_index: Index of this frame.
            simulation_time: Simulation time at this frame.

        Returns:
            Path to the saved PNG file.
        """
        data = self._extract_field_data(state)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.imshow(
            data.T,  # Transpose for correct orientation (x horizontal, y vertical)
            origin="lower",
            cmap=self.colormap,
            vmin=self.vmin,
            vmax=self.vmax,
            aspect="equal",
        )

        # Add vector arrows overlay if requested
        if self.show_vectors and self.plot_mode == "magnitude":
            vector_components = self._extract_vector_components(state)
            if vector_components is not None:
                u_data, v_data = vector_components
                ny, nx = data.shape

                # Subsample grid for arrows
                stride = max(1, min(nx, ny) // self.vector_density)
                x_indices = np.arange(0, nx, stride)
                y_indices = np.arange(0, ny, stride)
                X, Y = np.meshgrid(x_indices, y_indices)

                # Subsample vector components (note: data is transposed for display)
                U = u_data[::stride, ::stride].T
                V = v_data[::stride, ::stride].T

                # Draw quiver with white arrows and black edges for visibility
                ax.quiver(
                    X, Y, U, V,
                    color="white",
                    edgecolor="black",
                    linewidth=0.5,
                    scale=None,  # Auto-scale
                    alpha=0.8,
                )

        ax.axis("off")

        # Save
        frame_path = self.frames_dir / f"{frame_index:06d}.png"
        fig.savefig(frame_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # Track frame
        self.saved_frames.append({
            "frameIndex": frame_index,
            "simulationTime": simulation_time,
        })

        return frame_path

    def save_metadata(self, metadata: dict[str, Any]) -> Path:
        """Save metadata JSON.

        Args:
            metadata: Metadata dictionary to save.

        Returns:
            Path to the saved metadata file.
        """
        # Add colorbar bounds for interpreting the color scale
        if "visualization" not in metadata:
            metadata["visualization"] = {}

        metadata["visualization"]["colorbarMin"] = (
            self.vmin if self.vmin is not None else 0.0
        )
        metadata["visualization"]["colorbarMax"] = (
            self.vmax if self.vmax is not None else 1.0
        )

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
    ) -> Path:
        """Save the full trajectory as a numpy array.

        Saves a .npy file containing the trajectory with shape:
        - For scalar fields: (num_frames, resolution_x, resolution_y)
        - For multi-field systems: (num_frames, resolution_x, resolution_y)
          (only the selected field is saved)

        Also saves a times array with shape (num_frames,).

        Args:
            states: List of field states from the simulation.
            times: List of simulation times corresponding to each state.

        Returns:
            Path to the saved .npz file.
        """
        # Extract field data for each state
        trajectory_data = np.stack([
            self._extract_field_data(state) for state in states
        ])

        times_array = np.array(times)

        # Save as compressed npz with both trajectory and times
        array_path = self.output_dir / "trajectory.npz"
        np.savez_compressed(array_path, trajectory=trajectory_data, times=times_array)

        return array_path


def create_metadata(
    sim_id: str,
    preset_name: str,
    preset_metadata: Any,
    config: Any,
    total_time: float,
    frame_annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a complete metadata dictionary.

    Args:
        sim_id: Unique simulation ID.
        preset_name: Name of the PDE preset.
        preset_metadata: PDEMetadata from the preset.
        config: SimulationConfig object.
        total_time: Total simulation time.
        frame_annotations: List of per-frame annotations.

    Returns:
        Complete metadata dictionary.
    """
    # Load detailed description from markdown file
    description = get_description(preset_name)

    return {
        "id": sim_id,
        "preset": preset_name,
        "description": description,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generatorVersion": "1.0.0",
        "equations": preset_metadata.equations,
        "boundaryConditions": {
            "x": config.bc.x,
            "y": config.bc.y,
        },
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
            "timestepsPerFrame": config.output.frames_per_save,
            "totalTime": total_time,
            "resolution": [config.resolution, config.resolution],
        },
        "visualization": {
            "colormap": config.output.colormap,
            "whatToPlot": config.output.field_to_plot or preset_metadata.field_names[0],
        },
        "interventions": [],
        "frameAnnotations": frame_annotations,
    }

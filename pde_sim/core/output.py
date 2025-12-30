"""Output management for simulation frames and metadata."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pde import FieldCollection, ScalarField


class OutputManager:
    """Manages simulation output (frames and metadata).

    Handles saving PNG frames and JSON metadata for each simulation.
    """

    def __init__(
        self,
        base_path: Path | str,
        sim_id: str,
        colormap: str = "turbo",
        field_to_plot: str | None = None,
        dpi: int = 128,
        figsize: tuple[int, int] = (8, 8),
    ):
        """Initialize the output manager.

        Args:
            base_path: Base directory for output.
            sim_id: Unique simulation identifier.
            colormap: Matplotlib colormap name.
            field_to_plot: Name of field to render (for multi-field systems).
            dpi: DPI for saved images.
            figsize: Figure size in inches.
        """
        self.base_path = Path(base_path)
        self.sim_id = sim_id
        self.colormap = colormap
        self.field_to_plot = field_to_plot
        self.dpi = dpi
        self.figsize = figsize

        # Create output directories
        self.output_dir = self.base_path / self.sim_id
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # Track min/max values for consistent colormapping
        self.vmin: float | None = None
        self.vmax: float | None = None

        # Track saved frames
        self.saved_frames: list[dict[str, Any]] = []

    def _extract_field_data(
        self, state: ScalarField | FieldCollection
    ) -> np.ndarray:
        """Extract the numpy array from a field state.

        Args:
            state: The field state to extract data from.

        Returns:
            2D numpy array of field values.
        """
        if isinstance(state, FieldCollection):
            if self.field_to_plot:
                # Find field by label
                for field in state:
                    if hasattr(field, "label") and field.label == self.field_to_plot:
                        return field.data
                # If not found by label, try index
                try:
                    idx = int(self.field_to_plot)
                    return state[idx].data
                except (ValueError, IndexError):
                    pass
            # Default to first field
            return state[0].data
        else:
            return state.data

    def save_frame(
        self,
        state: ScalarField | FieldCollection,
        frame_index: int,
        simulation_time: float,
        update_range: bool = True,
    ) -> Path:
        """Save a single frame as PNG.

        Args:
            state: The field state to save.
            frame_index: Index of this frame.
            simulation_time: Simulation time at this frame.
            update_range: Whether to update global min/max range.

        Returns:
            Path to the saved PNG file.
        """
        data = self._extract_field_data(state)

        # Update global min/max for consistent coloring
        if update_range:
            current_min = float(np.min(data))
            current_max = float(np.max(data))

            if self.vmin is None:
                self.vmin = current_min
                self.vmax = current_max
            else:
                self.vmin = min(self.vmin, current_min)
                self.vmax = max(self.vmax, current_max)

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
        # Add visualization bounds
        if "visualization" not in metadata:
            metadata["visualization"] = {}

        metadata["visualization"]["minValue"] = (
            str(self.vmin) if self.vmin is not None else "0.0"
        )
        metadata["visualization"]["maxValue"] = (
            str(self.vmax) if self.vmax is not None else "1.0"
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
    return {
        "id": sim_id,
        "preset": preset_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generatorVersion": "1.0.0",
        "equations": {
            "reaction": list(preset_metadata.equations.values()),
            "diffusion": [f"D_{name}" for name in preset_metadata.field_names],
            "boundaryConditions": [config.bc.x] * preset_metadata.num_fields,
            "initialConditions": [config.init.type] * preset_metadata.num_fields,
        },
        "parameters": {
            "kinetic": config.parameters,
            "dt": config.dt,
            "spatialStep": config.domain_size / config.resolution,
            "domainScale": str(config.domain_size),
            "timesteppingScheme": config.solver.title(),
            "numSpecies": preset_metadata.num_fields,
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

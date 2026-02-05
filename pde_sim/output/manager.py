"""Output orchestration (multiple formats, per-field ranges, metadata writing)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pde import FieldCollection, ScalarField

from .handlers import OutputHandler, create_output_handler


class OutputManager:
    """Manages simulation output (frames and metadata)."""

    def __init__(
        self,
        base_path: Path | str,
        folder_name: str,
        *,
        colormap: str = "turbo",
        field_configs: list[tuple[str, str]] | None = None,
        dpi: int = 128,
        figsize: tuple[int, int] = (8, 8),
        output_formats: list[str] | None = None,
        fps: int = 30,
        ndim: int = 2,
        expected_num_frames: int | None = None,
    ) -> None:
        self.base_path = Path(base_path)
        self.folder_name = folder_name
        self.colormap = colormap
        self.dpi = dpi
        self.figsize = figsize
        self.output_formats = output_formats if output_formats is not None else ["png"]
        self.fps = fps
        self.ndim = ndim
        self.expected_num_frames = expected_num_frames

        self.field_configs = field_configs or []

        self.field_ranges: dict[str, tuple[float, float]] = {}
        self.field_colormaps: dict[str, str] = {n: c for n, c in self.field_configs}

        self.output_dir = self.base_path / self.folder_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.handlers: list[OutputHandler] = []
        for fmt in self.output_formats:
            handler = create_output_handler(fmt, fps=fps, ndim=ndim)
            handler.initialize(
                self.output_dir,
                self.field_configs,
                dpi,
                figsize,
                expected_num_frames=expected_num_frames,
            )
            self.handlers.append(handler)

        self.saved_frames: list[dict[str, Any]] = []

    def compute_range_for_field(
        self,
        states: list[ScalarField | FieldCollection],
        field_name: str,
    ) -> tuple[float, float]:
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
        if isinstance(state, FieldCollection):
            data = self._get_field_by_name(state, field_name)
            if data is None:
                data = state[0].data
        else:
            data = state.data

        if np.iscomplexobj(data):
            data = np.abs(data)
        return np.asarray(data)

    def _get_field_by_name(
        self, state: FieldCollection, field_name: str
    ) -> np.ndarray | None:
        for field in state:
            if hasattr(field, "label") and field.label == field_name:
                return field.data
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
        data = self._extract_field_data(state, field_name)
        colormap = self.field_colormaps.get(field_name, self.colormap)
        vmin, vmax = self.field_ranges.get(field_name, (None, None))

        for handler in self.handlers:
            handler.save_frame(
                data,
                field_name,
                frame_index,
                simulation_time,
                vmin,
                vmax,
                colormap,
            )

    def save_all_fields(
        self,
        state: ScalarField | FieldCollection,
        frame_index: int,
        simulation_time: float,
    ) -> None:
        for field_name, _cmap in self.field_configs:
            self.save_frame(state, field_name, frame_index, simulation_time)

        self.saved_frames.append(
            {"frameIndex": frame_index, "simulationTime": simulation_time}
        )

    def finalize(self) -> dict[str, Any]:
        result: dict[str, Any] = {"formats": self.output_formats}
        for handler in self.handlers:
            handler_metadata = handler.finalize()
            fmt = handler_metadata.get("format", "unknown")
            result[fmt] = handler_metadata
        return result

    def save_metadata(self, metadata: dict[str, Any]) -> Path:
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

        metadata["output"] = self.finalize()

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return metadata_path

    def get_frame_annotations(self) -> list[dict[str, Any]]:
        return self.saved_frames.copy()


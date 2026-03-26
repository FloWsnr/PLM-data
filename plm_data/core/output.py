"""Output handlers for simulation data."""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from dolfinx import fem

from plm_data.core.config import SimulationConfig
from plm_data.core.formats.gif_writer import GifWriter
from plm_data.core.formats.numpy_writer import NumpyWriter
from plm_data.core.formats.video_writer import VideoWriter
from plm_data.core.formats.vtk_writer import VTKWriter
from plm_data.core.interpolation import InterpolationCache, function_to_grid
from plm_data.core.logging import get_logger
from plm_data.presets.metadata import ConcreteOutputSpec, OutputSpec, PresetSpec

GridWriter = NumpyWriter | GifWriter | VideoWriter
FEMWriter = VTKWriter


@dataclass(frozen=True)
class GridOutputGroup:
    """One logical output expanded into one or more concrete grid arrays."""

    output_name: str
    output_spec: OutputSpec
    concrete_outputs: list[ConcreteOutputSpec]


@dataclass
class OutputTimingStats:
    """Accumulated timing breakdown for output generation."""

    write_frame_seconds: float = 0.0
    vtk_write_seconds: float = 0.0
    vector_view_seconds: float = 0.0
    grid_interpolation_seconds: float = 0.0
    grid_dispatch_seconds: float = 0.0
    writer_finalize_seconds: float = 0.0
    grid_interpolation_calls: int = 0
    format_finalize_seconds: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | int | dict[str, float]]:
        """Return a JSON-serializable timing summary."""
        return asdict(self)


class FrameWriter:
    """Capture simulation snapshots and delegate to format-specific writers."""

    def __init__(self, output_dir: Path, config: SimulationConfig, spec: PresetSpec):
        self.output_dir = output_dir
        self.config = config
        self.spec = spec
        self.frame_count = 0
        self.frame_times: list[float] = []
        self._logger = get_logger("output")
        self._resolution = tuple(self.config.output.resolution)
        self._timings = OutputTimingStats()

        self._interp_cache: InterpolationCache | None = None
        self._vector_vis_cache: dict[str, fem.Function] = {}

        output_modes = {
            output_name: selection.mode
            for output_name, selection in self.config.output.fields.items()
        }
        self._expected_outputs = self.spec.expected_outputs(
            output_modes=output_modes,
            gdim=self.config.domain.dimension,
        )
        self._output_specs = self.spec.outputs
        self.field_names = [output.name for output in self._expected_outputs]
        self._expected_base_fields = {
            output.source_name for output in self._expected_outputs
        }
        self._grid_output_groups = [
            GridOutputGroup(
                output_name=output_name,
                output_spec=output_spec,
                concrete_outputs=[
                    output
                    for output in self._expected_outputs
                    if output.output_name == output_name
                ],
            )
            for output_name, output_spec in self._output_specs.items()
            if self.config.output_mode(output_name) != "hidden"
        ]
        self._selected_vtk_outputs = {
            output_name: output_spec
            for output_name, output_spec in self._output_specs.items()
            if self.config.output_mode(output_name) != "hidden"
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)

        formats = set(config.output.formats)
        self._grid_writers: list[GridWriter] = []
        self._fem_writers: list[FEMWriter] = []

        if "numpy" in formats:
            self._grid_writers.append(NumpyWriter(output_dir, self.field_names))
        if "gif" in formats:
            self._grid_writers.append(GifWriter(output_dir))
        if "video" in formats:
            self._grid_writers.append(VideoWriter(output_dir))
        if "vtk" in formats:
            self._fem_writers.append(VTKWriter(output_dir))

        self._logger.info("  Output formats: %s", ", ".join(config.output.formats))

    def _vector_output_view(
        self,
        output_name: str,
        output_spec: OutputSpec,
        func: fem.Function,
    ) -> fem.Function:
        """Return a vector field in a component-addressable output space."""
        labels = output_spec.component_labels(self.config.domain.dimension)

        if func.function_space.num_sub_spaces == len(labels):
            return func

        if output_name not in self._vector_vis_cache:
            element = func.function_space.element
            degree = 1
            if getattr(element, "basix_element", None) is not None:
                degree = max(1, element.basix_element.degree)

            W = fem.functionspace(
                func.function_space.mesh,
                ("Discontinuous Lagrange", degree, (len(labels),)),
            )
            vis_func = fem.Function(W, name=f"{output_name}_vis")
            self._vector_vis_cache[output_name] = vis_func

        vis_func = self._vector_vis_cache[output_name]
        vis_func.interpolate(func)
        return vis_func

    def _validate_output_fields(self, fields: dict[str, fem.Function]) -> None:
        """Validate that all required base output fields are available."""
        missing = self._expected_base_fields - set(fields)
        if missing:
            raise ValueError(
                f"Output fields missing: {sorted(missing)}. "
                f"Expected at least {sorted(self._expected_base_fields)}, "
                f"got {sorted(fields)}."
            )

    def _selected_vtk_fields(
        self,
        fields: dict[str, fem.Function],
    ) -> dict[str, fem.Function]:
        selected: dict[str, fem.Function] = {}
        for output_name, output_spec in self._selected_vtk_outputs.items():
            source_name = output_spec.source_name
            if source_name not in fields:
                raise ValueError(
                    f"Output source '{source_name}' for '{output_name}' was not "
                    f"provided by the runtime problem."
                )
            selected[output_name] = fields[source_name]
        return selected

    def _dispatch_grid_field(self, name: str, arr: np.ndarray, t: float) -> None:
        """Send one concrete grid array to all grid writers."""
        dispatch_start = time.perf_counter()
        for writer in self._grid_writers:
            writer.on_frame_field(name, arr, t)
        self._timings.grid_dispatch_seconds += time.perf_counter() - dispatch_start

    def timing_summary(self) -> dict[str, float | int | dict[str, float]]:
        """Return output timing counters for logs, metadata, and runner summaries."""
        summary = self._timings.as_dict()
        summary["frame_count"] = self.frame_count
        return summary

    def write_frame(self, fields: dict[str, fem.Function], t: float) -> None:
        """Capture a snapshot of one or more output fields."""
        frame_start = time.perf_counter()

        vtk_fields = self._selected_vtk_fields(fields)
        if self._fem_writers:
            vtk_start = time.perf_counter()
            for writer in self._fem_writers:
                writer.on_frame(vtk_fields, t)
            self._timings.vtk_write_seconds += time.perf_counter() - vtk_start

        if self._grid_writers:
            self._validate_output_fields(fields)

            for output_group in self._grid_output_groups:
                base_function = fields[output_group.output_spec.source_name]

                if output_group.output_spec.shape == "scalar":
                    interp_start = time.perf_counter()
                    arr, self._interp_cache = function_to_grid(
                        base_function,
                        resolution=self._resolution,
                        cache=self._interp_cache,
                    )
                    self._timings.grid_interpolation_seconds += (
                        time.perf_counter() - interp_start
                    )
                    self._timings.grid_interpolation_calls += 1
                    self._dispatch_grid_field(
                        output_group.concrete_outputs[0].name,
                        arr,
                        t,
                    )
                    continue

                vector_start = time.perf_counter()
                view_func = self._vector_output_view(
                    output_group.output_name,
                    output_group.output_spec,
                    base_function,
                )
                self._timings.vector_view_seconds += time.perf_counter() - vector_start

                interp_start = time.perf_counter()
                grid, self._interp_cache = function_to_grid(
                    view_func,
                    resolution=self._resolution,
                    cache=self._interp_cache,
                )
                self._timings.grid_interpolation_seconds += (
                    time.perf_counter() - interp_start
                )
                self._timings.grid_interpolation_calls += 1

                if grid.ndim != len(self._resolution) + 1:
                    raise ValueError(
                        f"Vector output '{output_group.output_name}' produced grid "
                        f"shape {grid.shape}; expected component-first output."
                    )
                if grid.shape[0] != len(output_group.concrete_outputs):
                    raise ValueError(
                        f"Vector output '{output_group.output_name}' produced "
                        f"{grid.shape[0]} components, expected "
                        f"{len(output_group.concrete_outputs)}."
                    )

                for concrete_output, component_grid in zip(
                    output_group.concrete_outputs, grid
                ):
                    self._dispatch_grid_field(concrete_output.name, component_grid, t)

        self.frame_times.append(t)
        self.frame_count += 1
        self._timings.write_frame_seconds += time.perf_counter() - frame_start
        self._logger.debug("  Frame %d captured at t=%.6g", self.frame_count, t)

    def finalize(self) -> None:
        """Delegate finalization to all format writers and save metadata."""
        for writer in self._grid_writers:
            writer_name = type(writer).__name__.removesuffix("Writer").lower()
            finalize_start = time.perf_counter()
            writer.finalize()
            elapsed = time.perf_counter() - finalize_start
            self._timings.writer_finalize_seconds += elapsed
            self._timings.format_finalize_seconds[writer_name] = elapsed
        for writer in self._fem_writers:
            writer_name = type(writer).__name__.removesuffix("Writer").lower()
            finalize_start = time.perf_counter()
            writer.finalize()
            elapsed = time.perf_counter() - finalize_start
            self._timings.writer_finalize_seconds += elapsed
            self._timings.format_finalize_seconds[writer_name] = elapsed

        self._logger.info(
            "  Saved %d frames (%s) to %s",
            self.frame_count,
            ", ".join(self.field_names),
            self.output_dir,
        )
        self._logger.info(
            "  Output timing: frames=%.3fs, vtk=%.3fs, vector_view=%.3fs, "
            "interpolation=%.3fs, grid_dispatch=%.3fs, finalize=%.3fs",
            self._timings.write_frame_seconds,
            self._timings.vtk_write_seconds,
            self._timings.vector_view_seconds,
            self._timings.grid_interpolation_seconds,
            self._timings.grid_dispatch_seconds,
            self._timings.writer_finalize_seconds,
        )
        if self._timings.format_finalize_seconds:
            format_breakdown = ", ".join(
                f"{name}={seconds:.3f}s"
                for name, seconds in self._timings.format_finalize_seconds.items()
            )
            self._logger.info("  Finalize by format: %s", format_breakdown)

        metadata = {
            "num_frames": self.frame_count,
            "times": self.frame_times,
            "field_names": self.field_names,
            "output_resolution": self.config.output.resolution,
            "domain_type": self.config.domain.type,
            "domain_params": self.config.domain.params,
            "expected_outputs": [
                {
                    "name": output.name,
                    "output_name": output.output_name,
                    "source_name": output.source_name,
                    "component": output.component,
                }
                for output in self._expected_outputs
            ],
            "timings": self.timing_summary(),
        }
        with open(self.output_dir / "frames_meta.json", "w") as f:
            json.dump(metadata, f, indent=2)

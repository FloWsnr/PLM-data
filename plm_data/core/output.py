"""Output handlers for simulation data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from dolfinx import fem

from plm_data.core.config import SimulationConfig
from plm_data.core.formats.gif_writer import GifWriter
from plm_data.core.formats.numpy_writer import NumpyWriter
from plm_data.core.formats.video_writer import VideoWriter
from plm_data.core.formats.vtk_writer import VTKWriter
from plm_data.core.interpolation import InterpolationCache, function_to_array
from plm_data.core.logging import get_logger
from plm_data.presets.metadata import PresetSpec

# Type aliases for format writer categories
GridWriter = NumpyWriter | GifWriter | VideoWriter
FEMWriter = VTKWriter


class FrameWriter:
    """Capture simulation snapshots and delegate to format-specific writers."""

    def __init__(self, output_dir: Path, config: SimulationConfig, spec: PresetSpec):
        self.output_dir = output_dir
        self.config = config
        self.spec = spec
        self.frame_count = 0
        self.frame_times: list[float] = []
        self._logger = get_logger("output")

        # Grid interpolation state (only needed for numpy/gif/video)
        self._interp_cache: InterpolationCache | None = None
        self._vector_cache: dict[str, tuple[list[fem.Function], list[np.ndarray]]] = {}
        self._vector_vis_cache: dict[
            str, tuple[fem.Function, list[fem.Function], list[np.ndarray]]
        ] = {}

        output_modes = {
            field_name: field_config.output.mode
            for field_name, field_config in self.config.fields.items()
        }
        self._expected_outputs = self.spec.expected_outputs(
            output_modes=output_modes,
            gdim=self.config.domain.dimension,
        )
        self.field_names = [output.name for output in self._expected_outputs]
        self._expected_base_fields = {
            output.field_name for output in self._expected_outputs
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate format writers
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
        field_name: str,
        func: fem.Function,
    ) -> fem.Function:
        """Return a vector field in a component-addressable output space."""
        field_spec = self.spec.fields[field_name]
        labels = field_spec.component_labels(self.config.domain.dimension)

        if func.function_space.num_sub_spaces == len(labels):
            return func

        if field_name not in self._vector_vis_cache:
            element = func.function_space.element
            degree = 1
            if getattr(element, "basix_element", None) is not None:
                degree = max(1, element.basix_element.degree)

            W = fem.functionspace(
                func.function_space.mesh,
                ("Discontinuous Lagrange", degree, (len(labels),)),
            )
            vis_func = fem.Function(W, name=f"{field_name}_vis")

            component_funcs = []
            component_dofs: list[Any] = []
            for i, label in enumerate(labels):
                W_i, dofs_i = W.sub(i).collapse()
                component_funcs.append(fem.Function(W_i, name=f"{field_name}_{label}"))
                component_dofs.append(dofs_i)
            self._vector_vis_cache[field_name] = (
                vis_func,
                component_funcs,
                component_dofs,
            )

        vis_func, _, _ = self._vector_vis_cache[field_name]
        vis_func.interpolate(func)
        return vis_func

    def _split_vector_field(
        self,
        field_name: str,
        func: fem.Function,
    ) -> dict[str, fem.Function]:
        """Split a vector-valued function into scalar component functions."""
        field_spec = self.spec.fields[field_name]
        labels = field_spec.component_labels(self.config.domain.dimension)
        view_func = self._vector_output_view(field_name, func)

        if field_name not in self._vector_cache:
            component_funcs = []
            component_dofs: list[Any] = []
            for i, label in enumerate(labels):
                V_i, dofs_i = view_func.function_space.sub(i).collapse()
                component_funcs.append(fem.Function(V_i, name=f"{field_name}_{label}"))
                component_dofs.append(dofs_i)
            self._vector_cache[field_name] = (component_funcs, component_dofs)

        component_funcs, component_dofs = self._vector_cache[field_name]
        components: dict[str, fem.Function] = {}
        for label, component_func, dofs in zip(labels, component_funcs, component_dofs):
            component_func.x.array[:] = view_func.x.array[dofs]
            components[f"{field_name}_{label}"] = component_func
        return components

    def _expand_output_fields(
        self,
        fields: dict[str, fem.Function],
    ) -> dict[str, fem.Function]:
        """Expand base fields into concrete output arrays."""
        missing = self._expected_base_fields - set(fields)
        if missing:
            raise ValueError(
                f"Output fields missing: {sorted(missing)}. "
                f"Expected at least {sorted(self._expected_base_fields)}, "
                f"got {sorted(fields)}."
            )

        expanded: dict[str, fem.Function] = {}
        expanded_vectors: set[str] = set()
        for output in self._expected_outputs:
            field_spec = self.spec.fields[output.field_name]
            base_function = fields[output.field_name]
            if field_spec.shape == "scalar":
                expanded[output.name] = base_function
                continue

            if output.field_name not in expanded_vectors:
                expanded.update(
                    self._split_vector_field(output.field_name, base_function)
                )
                expanded_vectors.add(output.field_name)

        return {output.name: expanded[output.name] for output in self._expected_outputs}

    def write_frame(self, fields: dict[str, fem.Function], t: float) -> None:
        """Capture a snapshot of one or more base fields."""
        # FEM-native writers get raw base fields (before expansion/splitting)
        for writer in self._fem_writers:
            writer.on_frame(fields, t)

        # Grid-interpolated writers get numpy arrays
        if self._grid_writers:
            res = tuple(self.config.output.resolution)
            concrete_fields = self._expand_output_fields(fields)

            for name, func in concrete_fields.items():
                arr, self._interp_cache = function_to_array(
                    func,
                    resolution=res,
                    cache=self._interp_cache,
                )
                for writer in self._grid_writers:
                    writer.on_frame_field(name, arr, t)

        self.frame_times.append(t)
        self.frame_count += 1
        self._logger.debug("  Frame %d captured at t=%.6g", self.frame_count, t)

    def finalize(self) -> None:
        """Delegate finalization to all format writers and save metadata."""
        for writer in self._grid_writers:
            writer.finalize()
        for writer in self._fem_writers:
            writer.finalize()

        self._logger.info(
            "  Saved %d frames (%s) to %s",
            self.frame_count,
            ", ".join(self.field_names),
            self.output_dir,
        )

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
                    "field_name": output.field_name,
                    "component": output.component,
                }
                for output in self._expected_outputs
            ],
        }
        with open(self.output_dir / "frames_meta.json", "w") as f:
            json.dump(metadata, f, indent=2)

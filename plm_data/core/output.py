"""Output handlers for simulation data."""

import json
from pathlib import Path

import numpy as np
from dolfinx import fem

from plm_data.core.config import SimulationConfig
from plm_data.core.interpolation import InterpolationCache, function_to_array
from plm_data.core.logging import get_logger
from plm_data.presets.metadata import PresetSpec


class FrameWriter:
    """Capture simulation snapshots and write validated output arrays."""

    def __init__(self, output_dir: Path, config: SimulationConfig, spec: PresetSpec):
        self.output_dir = output_dir
        self.config = config
        self.spec = spec
        self.frame_count = 0
        self.frame_times: list[float] = []
        self._field_frames: dict[str, list[np.ndarray]] = {}
        self._interp_cache: InterpolationCache | None = None
        self._vector_cache: dict[str, tuple[list[fem.Function], list[np.ndarray]]] = {}
        self._logger = get_logger("output")

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

    def _split_vector_field(
        self,
        field_name: str,
        func: fem.Function,
    ) -> dict[str, fem.Function]:
        """Split a vector-valued function into scalar component functions."""
        field_spec = self.spec.fields[field_name]
        labels = field_spec.component_labels(self.config.domain.dimension)

        if field_name not in self._vector_cache:
            component_funcs = []
            component_dofs = []
            for i, label in enumerate(labels):
                V_i, dofs_i = func.function_space.sub(i).collapse()
                component_funcs.append(fem.Function(V_i, name=f"{field_name}_{label}"))
                component_dofs.append(dofs_i)
            self._vector_cache[field_name] = (component_funcs, component_dofs)

        component_funcs, component_dofs = self._vector_cache[field_name]
        components: dict[str, fem.Function] = {}
        for label, component_func, dofs in zip(labels, component_funcs, component_dofs):
            component_func.x.array[:] = func.x.array[dofs]
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
        res = tuple(self.config.output.resolution)
        concrete_fields = self._expand_output_fields(fields)

        for name, func in concrete_fields.items():
            arr, self._interp_cache = function_to_array(
                func,
                resolution=res,
                cache=self._interp_cache,
            )
            self._field_frames.setdefault(name, []).append(arr)

        self.frame_times.append(t)
        self.frame_count += 1
        self._logger.debug("  Frame %d captured at t=%.6g", self.frame_count, t)

    def finalize(self) -> None:
        """Stack all frames per field into a single array and save."""
        if "numpy" in self.config.output.formats:
            for name, frames in self._field_frames.items():
                np.save(self.output_dir / f"{name}.npy", np.stack(frames, axis=0))
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

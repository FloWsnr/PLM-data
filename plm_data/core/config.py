"""Simulation configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_COMPONENT_LABELS = ("x", "y", "z")


def _require(raw: dict[str, Any], key: str, context: str = "config") -> Any:
    """Require a key in a dict, raising a clear error if missing."""
    if key not in raw:
        raise ValueError(f"Missing required field '{key}' in {context}")
    return raw[key]


def _as_mapping(raw: Any, context: str) -> dict[str, Any]:
    """Require a mapping value."""
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a mapping. Got: {raw!r}")
    return raw


def _component_labels(gdim: int) -> tuple[str, ...]:
    """Return active vector component labels for the dimension."""
    return _COMPONENT_LABELS[:gdim]


def _infer_domain_dimension(domain_type: str, params: dict[str, Any]) -> int:
    """Infer the spatial dimension from the configured domain."""
    if domain_type == "rectangle":
        return 2
    if domain_type == "box":
        return 3
    size = params.get("size")
    if isinstance(size, list):
        return len(size)
    raise ValueError(
        f"Cannot infer spatial dimension for domain type '{domain_type}'. "
        "Provide a supported built-in domain."
    )


@dataclass
class FieldExpressionConfig:
    """Scalar or component-wise field value configuration."""

    type: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    components: dict[str, "FieldExpressionConfig"] = field(default_factory=dict)

    @property
    def is_componentwise(self) -> bool:
        """Return whether this expression is defined per component."""
        return bool(self.components)


@dataclass
class BoundaryConditionConfig:
    """Configuration for a single boundary condition."""

    type: str
    value: FieldExpressionConfig
    alpha: float | str | None = None


@dataclass
class DomainConfig:
    """Domain geometry configuration."""

    type: str
    params: dict[str, Any]

    @property
    def dimension(self) -> int:
        """Return the spatial dimension."""
        return _infer_domain_dimension(self.type, self.params)


@dataclass
class FieldOutputConfig:
    """Per-field output policy."""

    mode: str


@dataclass
class FieldConfig:
    """Configuration for one preset field."""

    boundary_conditions: dict[str, BoundaryConditionConfig] = field(
        default_factory=dict
    )
    source: FieldExpressionConfig | None = None
    initial_condition: FieldExpressionConfig | None = None
    output: FieldOutputConfig = field(
        default_factory=lambda: FieldOutputConfig("hidden")
    )


_VALID_FORMATS = {"numpy", "gif", "video", "vtk"}

_GRID_FORMATS = {"numpy", "gif", "video"}


@dataclass
class OutputConfig:
    """Output configuration."""

    path: Path
    resolution: list[int]
    num_frames: int
    formats: list[str]

    @property
    def needs_grid_interpolation(self) -> bool:
        """True if any format requires interpolated numpy arrays."""
        return bool(_GRID_FORMATS & set(self.formats))


@dataclass
class SolverConfig:
    """PETSc solver options (keys without prefix)."""

    options: dict[str, str]


@dataclass
class TimeConfig:
    """Time-stepping configuration."""

    dt: float
    t_end: float


@dataclass
class SimulationConfig:
    """Validated simulation configuration."""

    preset: str
    parameters: dict[str, float]
    domain: DomainConfig
    fields: dict[str, FieldConfig]
    output: OutputConfig
    solver: SolverConfig
    time: TimeConfig | None = None
    seed: int | None = None

    @property
    def dt(self) -> float | None:
        """Compatibility accessor for time step size."""
        if self.time is None:
            return None
        return self.time.dt

    @property
    def t_end(self) -> float | None:
        """Compatibility accessor for final time."""
        if self.time is None:
            return None
        return self.time.t_end

    @property
    def output_resolution(self) -> list[int]:
        """Return the configured output grid resolution."""
        return self.output.resolution

    def field(self, name: str) -> FieldConfig:
        """Return a configured field by name."""
        if name not in self.fields:
            raise KeyError(f"Unknown field '{name}'")
        return self.fields[name]


def _parse_scalar_expression(raw: Any, context: str) -> FieldExpressionConfig:
    """Parse a scalar field expression."""
    if isinstance(raw, (int, float)):
        return FieldExpressionConfig(type="constant", params={"value": raw})
    if isinstance(raw, str) and raw.startswith("param:"):
        return FieldExpressionConfig(type="constant", params={"value": raw})

    mapping = _as_mapping(raw, context)
    if "components" in mapping:
        raise ValueError(f"{context} must be scalar, not component-wise. Got: {raw!r}")
    expr_type = _require(mapping, "type", context)
    params = mapping.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"{context}.params must be a mapping. Got: {params!r}")
    return FieldExpressionConfig(type=expr_type, params=params)


def _parse_vector_expression(
    raw: Any,
    context: str,
    gdim: int,
) -> FieldExpressionConfig:
    """Parse a vector field expression."""
    labels = _component_labels(gdim)

    if isinstance(raw, list):
        if len(raw) != gdim:
            raise ValueError(
                f"{context} must have {gdim} components in {gdim}D. Got: {raw!r}"
            )
        components = {
            label: _parse_scalar_expression(value, f"{context}[{label}]")
            for label, value in zip(labels, raw)
        }
        return FieldExpressionConfig(components=components)

    mapping = _as_mapping(raw, context)

    if "components" in mapping:
        components_raw = _as_mapping(mapping["components"], f"{context}.components")
        if set(components_raw) != set(labels):
            raise ValueError(
                f"{context}.components must match {list(labels)} in {gdim}D. "
                f"Got {sorted(components_raw)}."
            )
        return FieldExpressionConfig(
            components={
                label: _parse_scalar_expression(
                    components_raw[label], f"{context}.components.{label}"
                )
                for label in labels
            }
        )

    expr_type = _require(mapping, "type", context)
    params = mapping.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"{context}.params must be a mapping. Got: {params!r}")
    if expr_type not in {"none", "zero", "custom"}:
        raise ValueError(
            f"{context} for a vector field must use 'components' or one of "
            f"['none', 'zero', 'custom']. Got type '{expr_type}'."
        )
    return FieldExpressionConfig(type=expr_type, params=params)


def _parse_field_expression(
    raw: Any,
    context: str,
    shape: str,
    gdim: int,
) -> FieldExpressionConfig:
    """Parse a field expression according to the declared field shape."""
    if shape == "scalar":
        return _parse_scalar_expression(raw, context)
    if shape == "vector":
        return _parse_vector_expression(raw, context, gdim)
    raise ValueError(f"Unknown field shape '{shape}' in {context}")


def _parse_boundary_condition(
    raw: Any,
    context: str,
    shape: str,
    gdim: int,
) -> BoundaryConditionConfig:
    """Parse one boundary condition config."""
    mapping = _as_mapping(raw, context)
    bc_type = _require(mapping, "type", context)
    bc_value = _parse_field_expression(
        _require(mapping, "value", context), f"{context}.value", shape, gdim
    )
    bc_alpha = mapping.get("alpha")
    if bc_type == "robin" and bc_alpha is None:
        raise ValueError(f"Robin BC in {context} requires 'alpha'")
    if shape != "scalar" and bc_type == "robin":
        raise ValueError(
            f"{context} uses BC type '{bc_type}', which is only supported for scalar fields"
        )
    return BoundaryConditionConfig(type=bc_type, value=bc_value, alpha=bc_alpha)


def _parse_output_config(raw: Any, context: str) -> FieldOutputConfig:
    """Parse a field output policy."""
    if isinstance(raw, str):
        return FieldOutputConfig(mode=raw)
    mapping = _as_mapping(raw, context)
    return FieldOutputConfig(mode=_require(mapping, "mode", context))


def load_config(path: str | Path) -> SimulationConfig:
    """Load and validate a simulation config from YAML."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    raw = _as_mapping(raw, "config")
    preset_name = _require(raw, "preset")
    parameters_raw = _as_mapping(_require(raw, "parameters"), "parameters")

    domain_raw = _as_mapping(_require(raw, "domain"), "domain")
    domain_type = _require(domain_raw, "type", "domain")
    domain_params = {k: v for k, v in domain_raw.items() if k != "type"}
    domain = DomainConfig(type=domain_type, params=domain_params)
    gdim = domain.dimension

    from plm_data.presets import get_preset

    preset = get_preset(preset_name)
    spec = preset.spec
    spec.validate_dimension(gdim)

    parameter_names = spec.parameter_names()
    if set(parameters_raw) != parameter_names:
        raise ValueError(
            f"Preset '{preset_name}' requires parameters {sorted(parameter_names)}. "
            f"Got {sorted(parameters_raw)}."
        )
    parameters = {name: float(value) for name, value in parameters_raw.items()}

    output_raw = _as_mapping(_require(raw, "output"), "output")
    output = OutputConfig(
        path=Path(_require(output_raw, "path", "output")),
        resolution=list(_require(output_raw, "resolution", "output")),
        num_frames=int(_require(output_raw, "num_frames", "output")),
        formats=list(_require(output_raw, "formats", "output")),
    )
    if len(output.resolution) != gdim:
        raise ValueError(
            f"output.resolution must have {gdim} entries in {gdim}D. "
            f"Got {output.resolution}."
        )
    if not output.formats:
        raise ValueError("output.formats must contain at least one format")
    invalid = set(output.formats) - _VALID_FORMATS
    if invalid:
        raise ValueError(
            f"Unknown output format(s) {sorted(invalid)}. "
            f"Valid formats: {sorted(_VALID_FORMATS)}."
        )
    if len(output.formats) != len(set(output.formats)):
        raise ValueError("output.formats contains duplicates")

    solver_raw = _as_mapping(_require(raw, "solver"), "solver")
    solver = SolverConfig(options={str(k): str(v) for k, v in solver_raw.items()})

    if spec.steady_state:
        if "time" in raw:
            raise ValueError(
                f"Preset '{preset_name}' is steady-state and cannot use 'time'"
            )
        time = None
    else:
        time_raw = _as_mapping(_require(raw, "time"), "time")
        time = TimeConfig(
            dt=float(_require(time_raw, "dt", "time")),
            t_end=float(_require(time_raw, "t_end", "time")),
        )

    fields_raw = _as_mapping(_require(raw, "fields"), "fields")
    if set(fields_raw) != set(spec.fields):
        raise ValueError(
            f"Preset '{preset_name}' requires fields {sorted(spec.fields)}. "
            f"Got {sorted(fields_raw)}."
        )

    fields: dict[str, FieldConfig] = {}
    for field_name, field_spec in spec.fields.items():
        context = f"fields.{field_name}"
        field_raw = _as_mapping(fields_raw[field_name], context)

        allowed_keys = {"output"}
        if field_spec.allow_boundary_conditions:
            allowed_keys.add("boundary_conditions")
        if field_spec.allow_source:
            allowed_keys.add("source")
        if field_spec.allow_initial_condition:
            allowed_keys.add("initial_condition")

        unexpected_keys = set(field_raw) - allowed_keys
        if unexpected_keys:
            raise ValueError(
                f"{context} has unsupported keys {sorted(unexpected_keys)}. "
                f"Allowed keys: {sorted(allowed_keys)}."
            )

        if field_spec.allow_boundary_conditions:
            boundary_raw = _as_mapping(
                _require(field_raw, "boundary_conditions", context),
                f"{context}.boundary_conditions",
            )
            boundary_conditions = {
                name: _parse_boundary_condition(
                    bc_raw,
                    f"{context}.boundary_conditions.{name}",
                    field_spec.shape,
                    gdim,
                )
                for name, bc_raw in boundary_raw.items()
            }
        else:
            boundary_conditions = {}

        if field_spec.allow_source:
            source = _parse_field_expression(
                _require(field_raw, "source", context),
                f"{context}.source",
                field_spec.shape,
                gdim,
            )
        else:
            source = None

        if field_spec.allow_initial_condition:
            initial_condition = _parse_field_expression(
                _require(field_raw, "initial_condition", context),
                f"{context}.initial_condition",
                field_spec.shape,
                gdim,
            )
        else:
            initial_condition = None

        output_policy = _parse_output_config(
            _require(field_raw, "output", context), f"{context}.output"
        )
        field_spec.validate_output_mode(output_policy.mode)

        fields[field_name] = FieldConfig(
            boundary_conditions=boundary_conditions,
            source=source,
            initial_condition=initial_condition,
            output=output_policy,
        )

    return SimulationConfig(
        preset=preset_name,
        parameters=parameters,
        domain=domain,
        fields=fields,
        output=output,
        solver=solver,
        time=time,
        seed=raw.get("seed"),
    )

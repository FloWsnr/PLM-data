"""Simulation configuration loading and validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_COMPONENT_LABELS = ("x", "y", "z")
_VALID_FORMATS = {"numpy", "gif", "video", "vtk"}
_GRID_FORMATS = {"numpy", "gif", "video"}


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
    builtin_dims = {
        "interval": 1,
        "rectangle": 2,
        "box": 3,
    }
    if domain_type in builtin_dims:
        return builtin_dims[domain_type]

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
    """Configuration for one boundary operator entry."""

    type: str
    value: FieldExpressionConfig | None = None
    pair_with: str | None = None
    operator_parameters: dict[str, Any] = field(default_factory=dict)
    alpha: float | str | None = None

    def __post_init__(self) -> None:
        if self.alpha is not None:
            self.operator_parameters = {
                **self.operator_parameters,
                "alpha": self.alpha,
            }


@dataclass
class BoundaryFieldConfig:
    """Configuration for all side conditions of one BC-addressable field."""

    sides: dict[str, list[BoundaryConditionConfig]] = field(default_factory=dict)

    def side_conditions(self, name: str) -> list[BoundaryConditionConfig]:
        """Return configured conditions for one side."""
        if name not in self.sides:
            raise KeyError(f"Unknown boundary side '{name}'")
        return self.sides[name]

    def periodic_pair_keys(self) -> set[frozenset[str]]:
        """Return all active periodic side pairs."""
        pairs: set[frozenset[str]] = set()
        for side, entries in self.sides.items():
            for entry in entries:
                if entry.type == "periodic":
                    if entry.pair_with is None:
                        raise ValueError(
                            f"Periodic boundary on side '{side}' is missing "
                            "'pair_with'."
                        )
                    pairs.add(frozenset({side, entry.pair_with}))
        return pairs

    @property
    def has_periodic(self) -> bool:
        """Return whether any side uses the periodic operator."""
        return bool(self.periodic_pair_keys())


@dataclass
class PeriodicMapConfig:
    """Declarative periodic map for a custom or imported domain."""

    slave: str
    master: str
    matrix: list[list[float]]
    offset: list[float]


@dataclass
class DomainConfig:
    """Domain geometry configuration."""

    type: str
    params: dict[str, Any]
    periodic_maps: dict[str, PeriodicMapConfig] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        """Return the spatial dimension."""
        return _infer_domain_dimension(self.type, self.params)


@dataclass
class OutputSelectionConfig:
    """Per-output selection policy."""

    mode: str


@dataclass
class InputConfig:
    """Configuration for one preset input."""

    source: FieldExpressionConfig | None = None
    initial_condition: FieldExpressionConfig | None = None


@dataclass
class OutputConfig:
    """Output configuration."""

    resolution: list[int]
    num_frames: int
    formats: list[str]
    fields: dict[str, OutputSelectionConfig]
    path: Path | None = None

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
    inputs: dict[str, InputConfig]
    boundary_conditions: dict[str, BoundaryFieldConfig]
    output: OutputConfig
    solver: SolverConfig
    time: TimeConfig | None = None
    seed: int | None = None
    coefficients: dict[str, FieldExpressionConfig] = field(default_factory=dict)

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

    def input(self, name: str) -> InputConfig:
        """Return a configured input by name."""
        if name not in self.inputs:
            raise KeyError(f"Unknown input '{name}'")
        return self.inputs[name]

    def coefficient(self, name: str) -> FieldExpressionConfig:
        """Return a configured coefficient by name."""
        if name not in self.coefficients:
            raise KeyError(f"Unknown coefficient '{name}'")
        return self.coefficients[name]

    def boundary_field(self, name: str) -> BoundaryFieldConfig:
        """Return configured boundary conditions for one BC field."""
        if name not in self.boundary_conditions:
            raise KeyError(f"Unknown boundary field '{name}'")
        return self.boundary_conditions[name]

    def field(self, name: str) -> InputConfig:
        """Compatibility accessor for input configs."""
        return self.input(name)

    def output_mode(self, name: str) -> str:
        """Return the configured output mode for a named output."""
        if name not in self.output.fields:
            raise KeyError(f"Unknown output '{name}'")
        return self.output.fields[name].mode

    @property
    def has_periodic_boundary_conditions(self) -> bool:
        """Return whether any BC field uses periodic side pairs."""
        return any(
            field_config.has_periodic
            for field_config in self.boundary_conditions.values()
        )


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
            for label, value in zip(labels, raw, strict=True)
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
    """Parse a field expression according to the declared shape."""
    if shape == "scalar":
        return _parse_scalar_expression(raw, context)
    if shape == "vector":
        return _parse_vector_expression(raw, context, gdim)
    raise ValueError(f"Unknown field shape '{shape}' in {context}")


def _parse_operator_parameters(
    raw: Any,
    context: str,
    allowed: tuple[str, ...],
) -> dict[str, Any]:
    """Parse operator-specific scalar parameters."""
    if not allowed:
        if raw is None:
            return {}
        mapping = _as_mapping(raw, context)
        if mapping:
            raise ValueError(
                f"{context} does not allow operator parameters. Got {sorted(mapping)}."
            )
        return {}

    if raw is None:
        raise ValueError(f"{context} must contain exactly {sorted(allowed)}. Got [].")

    mapping = _as_mapping(raw, context)
    if set(mapping) != set(allowed):
        raise ValueError(
            f"{context} must contain exactly {sorted(allowed)}. Got {sorted(mapping)}."
        )
    return dict(mapping)


def _parse_boundary_condition(
    raw: Any,
    context: str,
    shape: str,
    gdim: int,
    *,
    operators: dict[str, Any],
) -> BoundaryConditionConfig:
    """Parse one boundary operator config."""
    mapping = _as_mapping(raw, context)
    operator = _require(mapping, "operator", context)
    if operator not in operators:
        raise ValueError(
            f"{context} uses unsupported operator '{operator}'. "
            f"Allowed operators: {sorted(operators)}."
        )
    operator_spec = operators[operator]

    if operator_spec.value_shape is None:
        if "value" in mapping:
            raise ValueError(
                f"{context} operator '{operator}' does not accept a value."
            )
        value = None
    else:
        value_shape = (
            shape if operator_spec.value_shape == "field" else operator_spec.value_shape
        )
        value = _parse_field_expression(
            _require(mapping, "value", context),
            f"{context}.value",
            value_shape,
            gdim,
        )

    pair_with = mapping.get("pair_with")
    if operator_spec.requires_pair_with and pair_with is None:
        raise ValueError(f"{context} operator '{operator}' requires 'pair_with'.")
    if not operator_spec.requires_pair_with and pair_with is not None:
        raise ValueError(
            f"{context} operator '{operator}' does not accept 'pair_with'."
        )

    operator_parameters = _parse_operator_parameters(
        mapping.get("operator_parameters"),
        f"{context}.operator_parameters",
        operator_spec.operator_parameter_names,
    )

    unexpected_keys = set(mapping) - {
        "operator",
        "value",
        "pair_with",
        "operator_parameters",
    }
    if unexpected_keys:
        raise ValueError(f"{context} has unsupported keys {sorted(unexpected_keys)}.")

    return BoundaryConditionConfig(
        type=operator,
        value=value,
        pair_with=pair_with,
        operator_parameters=operator_parameters,
    )


def _parse_boundary_field(
    raw: Any,
    context: str,
    *,
    shape: str,
    gdim: int,
    operators: dict[str, Any],
) -> BoundaryFieldConfig:
    """Parse all side conditions for one BC-addressable field."""
    mapping = _as_mapping(raw, context)
    sides: dict[str, list[BoundaryConditionConfig]] = {}
    for side_name, entries_raw in mapping.items():
        side_context = f"{context}.{side_name}"
        if not isinstance(entries_raw, list):
            raise ValueError(f"{side_context} must be a list. Got {entries_raw!r}")
        if not entries_raw:
            raise ValueError(f"{side_context} must contain at least one operator.")
        sides[side_name] = [
            _parse_boundary_condition(
                entry_raw,
                f"{side_context}[{index}]",
                shape,
                gdim,
                operators=operators,
            )
            for index, entry_raw in enumerate(entries_raw)
        ]

    for side_name, entries in sides.items():
        periodic_entries = [entry for entry in entries if entry.type == "periodic"]
        if not periodic_entries:
            continue
        if len(entries) != 1:
            raise ValueError(
                f"{context}.{side_name} cannot mix 'periodic' with other operators."
            )
        pair_with = periodic_entries[0].pair_with
        if pair_with not in sides:
            raise ValueError(
                f"{context}.{side_name} pairs with unknown side '{pair_with}'."
            )
        paired_entries = sides[pair_with]
        if len(paired_entries) != 1 or paired_entries[0].type != "periodic":
            raise ValueError(
                f"{context}.{side_name} periodic pair must be reciprocal with "
                f"'{pair_with}', and the paired side must also be a pure "
                "periodic entry."
            )
        if paired_entries[0].pair_with != side_name:
            raise ValueError(
                f"{context}.{side_name} periodic pair must be reciprocal with "
                f"'{pair_with}'."
            )

    return BoundaryFieldConfig(sides=sides)


def _parse_output_selection(raw: Any, context: str) -> OutputSelectionConfig:
    """Parse an output selection policy."""
    if isinstance(raw, str):
        return OutputSelectionConfig(mode=raw)
    mapping = _as_mapping(raw, context)
    return OutputSelectionConfig(mode=_require(mapping, "mode", context))


def _parse_periodic_map(raw: Any, context: str, gdim: int) -> PeriodicMapConfig:
    """Parse one domain-level periodic map declaration."""
    mapping = _as_mapping(raw, context)
    slave = _require(mapping, "slave", context)
    master = _require(mapping, "master", context)
    transform = _as_mapping(
        _require(mapping, "transform", context), f"{context}.transform"
    )
    transform_type = _require(transform, "type", f"{context}.transform")
    if transform_type != "affine":
        raise ValueError(
            f"{context}.transform.type must be 'affine'. Got '{transform_type}'."
        )
    matrix = _require(transform, "matrix", f"{context}.transform")
    offset = _require(transform, "offset", f"{context}.transform")
    if not isinstance(matrix, list) or len(matrix) != gdim:
        raise ValueError(
            f"{context}.transform.matrix must have {gdim} rows in {gdim}D. "
            f"Got {matrix!r}."
        )
    parsed_matrix: list[list[float]] = []
    for row_index, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != gdim:
            raise ValueError(
                f"{context}.transform.matrix[{row_index}] must have {gdim} "
                f"entries in {gdim}D. Got {row!r}."
            )
        parsed_matrix.append([float(value) for value in row])
    if not isinstance(offset, list) or len(offset) != gdim:
        raise ValueError(
            f"{context}.transform.offset must have {gdim} entries in {gdim}D. "
            f"Got {offset!r}."
        )
    unexpected_keys = set(mapping) - {"slave", "master", "transform"}
    if unexpected_keys:
        raise ValueError(f"{context} has unsupported keys {sorted(unexpected_keys)}.")
    return PeriodicMapConfig(
        slave=str(slave),
        master=str(master),
        matrix=parsed_matrix,
        offset=[float(value) for value in offset],
    )


def _parse_periodic_maps(
    raw: Any, context: str, gdim: int
) -> dict[str, PeriodicMapConfig]:
    """Parse all domain-level periodic map declarations."""
    mapping = _as_mapping(raw, context)
    return {
        name: _parse_periodic_map(periodic_raw, f"{context}.{name}", gdim)
        for name, periodic_raw in mapping.items()
    }


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
    domain_params = {
        key: value
        for key, value in domain_raw.items()
        if key not in {"type", "periodic_maps"}
    }
    gdim = _infer_domain_dimension(domain_type, domain_params)
    periodic_maps_raw = domain_raw.get("periodic_maps", {})
    periodic_maps = _parse_periodic_maps(
        periodic_maps_raw,
        "domain.periodic_maps",
        gdim,
    )
    domain = DomainConfig(
        type=domain_type,
        params=domain_params,
        periodic_maps=periodic_maps,
    )
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

    if spec.coefficients:
        coefficients_raw = _as_mapping(
            _require(raw, "coefficients", "config"), "coefficients"
        )
        if set(coefficients_raw) != set(spec.coefficients):
            raise ValueError(
                f"Preset '{preset_name}' requires coefficients "
                f"{sorted(spec.coefficients)}. Got {sorted(coefficients_raw)}."
            )
        coefficients = {
            coefficient_name: _parse_field_expression(
                coefficients_raw[coefficient_name],
                f"coefficients.{coefficient_name}",
                coefficient_spec.shape,
                gdim,
            )
            for coefficient_name, coefficient_spec in spec.coefficients.items()
        }
    else:
        coefficients_raw = raw.get("coefficients")
        if coefficients_raw not in (None, {}):
            raise ValueError(f"Preset '{preset_name}' does not support coefficients.")
        coefficients = {}

    output_raw = _as_mapping(_require(raw, "output"), "output")
    unexpected_output_keys = set(output_raw) - {
        "resolution",
        "num_frames",
        "formats",
        "fields",
    }
    if unexpected_output_keys:
        raise ValueError(
            f"output has unsupported keys {sorted(unexpected_output_keys)}."
        )
    output_fields_raw = _as_mapping(
        _require(output_raw, "fields", "output"), "output.fields"
    )
    if set(output_fields_raw) != set(spec.outputs):
        raise ValueError(
            f"Preset '{preset_name}' requires outputs {sorted(spec.outputs)}. "
            f"Got {sorted(output_fields_raw)}."
        )
    output_fields = {
        output_name: _parse_output_selection(
            output_fields_raw[output_name], f"output.fields.{output_name}"
        )
        for output_name in spec.outputs
    }
    for output_name, output_spec in spec.outputs.items():
        output_spec.validate_output_mode(output_fields[output_name].mode)

    output = OutputConfig(
        path=None,
        resolution=list(_require(output_raw, "resolution", "output")),
        num_frames=int(_require(output_raw, "num_frames", "output")),
        formats=list(_require(output_raw, "formats", "output")),
        fields=output_fields,
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

    inputs_raw = _as_mapping(_require(raw, "inputs"), "inputs")
    if set(inputs_raw) != set(spec.inputs):
        raise ValueError(
            f"Preset '{preset_name}' requires inputs {sorted(spec.inputs)}. "
            f"Got {sorted(inputs_raw)}."
        )

    inputs: dict[str, InputConfig] = {}
    for input_name, input_spec in spec.inputs.items():
        context = f"inputs.{input_name}"
        input_raw = _as_mapping(inputs_raw[input_name], context)

        allowed_keys: set[str] = set()
        if input_spec.allow_source:
            allowed_keys.add("source")
        if input_spec.allow_initial_condition:
            allowed_keys.add("initial_condition")

        unexpected_keys = set(input_raw) - allowed_keys
        if unexpected_keys:
            raise ValueError(
                f"{context} has unsupported keys {sorted(unexpected_keys)}. "
                f"Allowed keys: {sorted(allowed_keys)}."
            )

        if input_spec.allow_source:
            source = _parse_field_expression(
                _require(input_raw, "source", context),
                f"{context}.source",
                input_spec.shape,
                gdim,
            )
        else:
            source = None

        if input_spec.allow_initial_condition:
            initial_condition = _parse_field_expression(
                _require(input_raw, "initial_condition", context),
                f"{context}.initial_condition",
                input_spec.shape,
                gdim,
            )
        else:
            initial_condition = None

        inputs[input_name] = InputConfig(
            source=source,
            initial_condition=initial_condition,
        )

    if spec.boundary_fields:
        boundary_raw = _as_mapping(
            _require(raw, "boundary_conditions", "config"),
            "boundary_conditions",
        )
        if set(boundary_raw) != set(spec.boundary_fields):
            raise ValueError(
                f"Preset '{preset_name}' requires boundary condition fields "
                f"{sorted(spec.boundary_fields)}. Got {sorted(boundary_raw)}."
            )
        boundary_conditions = {
            field_name: _parse_boundary_field(
                boundary_raw[field_name],
                f"boundary_conditions.{field_name}",
                shape=field_spec.shape,
                gdim=gdim,
                operators=field_spec.operators,
            )
            for field_name, field_spec in spec.boundary_fields.items()
        }
    else:
        boundary_conditions_raw = raw.get("boundary_conditions")
        if boundary_conditions_raw not in (None, {}):
            raise ValueError(
                f"Preset '{preset_name}' does not support boundary_conditions."
            )
        boundary_conditions = {}

    return SimulationConfig(
        preset=preset_name,
        parameters=parameters,
        domain=domain,
        inputs=inputs,
        boundary_conditions=boundary_conditions,
        output=output,
        solver=solver,
        time=time,
        seed=raw.get("seed"),
        coefficients=coefficients,
    )

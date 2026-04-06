"""Simulation configuration loading and validation."""

from copy import deepcopy
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

from mpi4py import MPI
import yaml

from plm_data.core.solver_strategies import ALL_SOLVER_STRATEGIES

_COMPONENT_LABELS = ("x", "y", "z")
_VALID_FORMATS = {"numpy", "gif", "video", "vtk"}
_GRID_FORMATS = {"numpy", "gif", "video"}
_REF_KEY = "$ref"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRAGMENT_CATALOG_PATH = _REPO_ROOT / "configs" / "_fragments.yaml"
_INITIAL_CONDITION_EXPR_TYPES = {
    "none",
    "zero",
    "custom",
    "constant",
    "sine_product",
    "cosine_product",
    "gaussian_bump",
    "radial_cosine",
    "affine",
    "step",
    "gaussian_noise",
    "gaussian_blobs",
    "gaussian_wave_packet",
    "sine_waves",
    "quadrants",
}
_TOP_LEVEL_CONFIG_KEYS = {
    "preset",
    "parameters",
    "domain",
    "coefficients",
    "inputs",
    "boundary_conditions",
    "output",
    "solver",
    "time",
    "seed",
    "stochastic",
}
_VALID_STOCHASTIC_STATE_COUPLINGS = {
    "additive",
    "multiplicative_self",
    "saturating_self",
}
_VALID_STOCHASTIC_COEFFICIENT_MODES = {
    "additive",
    "multiplicative",
}


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


def _is_param_ref(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("param:")


def _validate_numeric_literal_or_param_ref(value: Any, context: str) -> None:
    if isinstance(value, (int, float)) or _is_param_ref(value):
        return
    raise ValueError(
        f"{context} must be a number or 'param:<name>' reference. Got {value!r}."
    )


def _validate_integer_literal_or_param_ref(value: Any, context: str) -> None:
    if isinstance(value, int):
        return
    if isinstance(value, float) and value.is_integer():
        return
    if _is_param_ref(value):
        return
    raise ValueError(
        f"{context} must be an integer or 'param:<name>' reference. Got {value!r}."
    )


def _resolve_numeric_literal_or_param_ref(
    value: Any,
    parameters: dict[str, float],
    context: str,
) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if _is_param_ref(value):
        parameter_name = value[len("param:") :]
        if parameter_name not in parameters:
            raise ValueError(
                f"{context} references unknown parameter '{parameter_name}'. "
                f"Available parameters: {sorted(parameters)}."
            )
        return float(parameters[parameter_name])
    raise ValueError(
        f"{context} must be a number or 'param:<name>' reference. Got {value!r}."
    )


def _validate_sampleable_numeric(
    value: Any,
    context: str,
) -> None:
    if not isinstance(value, dict) or "sample" not in value:
        _validate_numeric_literal_or_param_ref(value, context)
        return

    sample_type = _require(value, "sample", context)
    if sample_type == "uniform":
        if set(value) != {"sample", "min", "max"}:
            raise ValueError(
                f"{context} uniform sampler must contain exactly ['max', 'min', "
                f"'sample']. Got {sorted(value)}."
            )
        _validate_numeric_literal_or_param_ref(value["min"], f"{context}.min")
        _validate_numeric_literal_or_param_ref(value["max"], f"{context}.max")
        return

    if sample_type == "normal":
        if set(value) != {"sample", "mean", "std"}:
            raise ValueError(
                f"{context} normal sampler must contain exactly ['mean', 'sample', "
                f"'std']. Got {sorted(value)}."
            )
        _validate_numeric_literal_or_param_ref(value["mean"], f"{context}.mean")
        _validate_numeric_literal_or_param_ref(value["std"], f"{context}.std")
        return

    if sample_type == "randint":
        if set(value) != {"sample", "min", "max"}:
            raise ValueError(
                f"{context} randint sampler must contain exactly ['max', 'min', "
                f"'sample']. Got {sorted(value)}."
            )
        _validate_numeric_literal_or_param_ref(value["min"], f"{context}.min")
        _validate_numeric_literal_or_param_ref(value["max"], f"{context}.max")
        return

    raise ValueError(f"{context} uses unknown sampler '{sample_type}'.")


def _validate_sampleable_integer(
    value: Any,
    context: str,
) -> None:
    if not isinstance(value, dict) or "sample" not in value:
        _validate_integer_literal_or_param_ref(value, context)
        return
    _validate_sampleable_numeric(value, context)


def _validate_sampleable_vector(
    values: Any,
    context: str,
    *,
    gdim: int,
) -> None:
    if not isinstance(values, list) or len(values) != gdim:
        raise ValueError(
            f"{context} must have {gdim} entries in {gdim}D. Got {values!r}."
        )
    for index, value in enumerate(values):
        _validate_sampleable_numeric(value, f"{context}[{index}]")


def _validate_initial_condition_scalar_expression(
    expr,
    context: str,
    *,
    gdim: int,
) -> None:
    if expr.type not in _INITIAL_CONDITION_EXPR_TYPES:
        raise ValueError(
            f"{context} uses unsupported initial_condition type '{expr.type}'. "
            f"Allowed types: {sorted(_INITIAL_CONDITION_EXPR_TYPES)}."
        )

    params = expr.params
    expr_type = expr.type
    axis_frequency_keys = ("kx", "ky", "kz")[:gdim]
    coordinate_keys = {"x", "y", "z"} & set(_component_labels(gdim))

    if expr_type in {"none", "zero", "custom"}:
        if params not in ({}, None):
            raise ValueError(f"{context}.params must be empty for type '{expr_type}'.")
        return

    if expr_type == "constant":
        if set(params) != {"value"}:
            raise ValueError(
                f"{context}.params must contain exactly ['value'] for type "
                f"'{expr_type}'. Got {sorted(params)}."
            )
        _validate_sampleable_numeric(params["value"], f"{context}.params.value")
        return

    if expr_type in {"sine_product", "cosine_product"}:
        allowed_keys = {"amplitude", *axis_frequency_keys}
        if set(params) - allowed_keys:
            raise ValueError(
                f"{context}.params for type '{expr_type}' allows only "
                f"{sorted(allowed_keys)}. Got {sorted(params)}."
            )
        if "amplitude" not in params:
            raise ValueError(f"{context}.params requires 'amplitude'.")
        if not any(key in params for key in axis_frequency_keys):
            raise ValueError(
                f"{context}.params for type '{expr_type}' requires at least one of "
                f"{list(axis_frequency_keys)}."
            )
        for key, value in params.items():
            _validate_sampleable_numeric(value, f"{context}.params.{key}")
        return

    if expr_type == "gaussian_bump":
        if set(params) != {"amplitude", "sigma", "center"}:
            raise ValueError(
                f"{context}.params must contain exactly ['amplitude', 'center', "
                f"'sigma'] for type '{expr_type}'. Got {sorted(params)}."
            )
        _validate_sampleable_numeric(params["amplitude"], f"{context}.params.amplitude")
        _validate_sampleable_numeric(params["sigma"], f"{context}.params.sigma")
        _validate_sampleable_vector(
            params["center"],
            f"{context}.params.center",
            gdim=gdim,
        )
        return

    if expr_type == "radial_cosine":
        if set(params) != {"base", "amplitude", "frequency", "center"}:
            raise ValueError(
                f"{context}.params must contain exactly ['amplitude', 'base', "
                f"'center', 'frequency'] for type '{expr_type}'. Got "
                f"{sorted(params)}."
            )
        _validate_sampleable_numeric(params["base"], f"{context}.params.base")
        _validate_sampleable_numeric(params["amplitude"], f"{context}.params.amplitude")
        _validate_sampleable_numeric(params["frequency"], f"{context}.params.frequency")
        _validate_sampleable_vector(
            params["center"],
            f"{context}.params.center",
            gdim=gdim,
        )
        return

    if expr_type == "affine":
        allowed_keys = {"constant", *coordinate_keys}
        if set(params) - allowed_keys:
            raise ValueError(
                f"{context}.params for type '{expr_type}' allows only "
                f"{sorted(allowed_keys)}. Got {sorted(params)}."
            )
        for key, value in params.items():
            _validate_sampleable_numeric(value, f"{context}.params.{key}")
        return

    if expr_type == "step":
        if set(params) != {"value_left", "value_right", "x_split", "axis"}:
            raise ValueError(
                f"{context}.params must contain exactly ['axis', 'value_left', "
                f"'value_right', 'x_split'] for type '{expr_type}'. Got "
                f"{sorted(params)}."
            )
        _validate_sampleable_numeric(
            params["value_left"], f"{context}.params.value_left"
        )
        _validate_sampleable_numeric(
            params["value_right"], f"{context}.params.value_right"
        )
        _validate_sampleable_numeric(params["x_split"], f"{context}.params.x_split")
        _validate_sampleable_integer(params["axis"], f"{context}.params.axis")
        return

    if expr_type == "gaussian_noise":
        if set(params) != {"mean", "std"}:
            raise ValueError(
                f"{context}.params must contain exactly ['mean', 'std'] for type "
                f"'{expr_type}'. Got {sorted(params)}."
            )
        _validate_sampleable_numeric(params["mean"], f"{context}.params.mean")
        _validate_sampleable_numeric(params["std"], f"{context}.params.std")
        return

    if expr_type == "gaussian_blobs":
        if set(params) != {"background", "generators"}:
            raise ValueError(
                f"{context}.params must contain exactly ['background', "
                f"'generators'] for type '{expr_type}'. Got {sorted(params)}."
            )
        _validate_sampleable_numeric(
            params["background"], f"{context}.params.background"
        )
        generators = params["generators"]
        if not isinstance(generators, list) or not generators:
            raise ValueError(f"{context}.params.generators must be a non-empty list.")
        for index, generator in enumerate(generators):
            generator_context = f"{context}.params.generators[{index}]"
            generator_mapping = _as_mapping(generator, generator_context)
            expected_keys = {
                "count",
                "amplitude",
                "sigma",
                "center",
                "aspect_ratio",
            }
            if set(generator_mapping) != expected_keys:
                raise ValueError(
                    f"{generator_context} must contain exactly "
                    f"{sorted(expected_keys)}. Got {sorted(generator_mapping)}."
                )
            _validate_sampleable_integer(
                generator_mapping["count"], f"{generator_context}.count"
            )
            _validate_sampleable_numeric(
                generator_mapping["amplitude"], f"{generator_context}.amplitude"
            )
            _validate_sampleable_numeric(
                generator_mapping["sigma"], f"{generator_context}.sigma"
            )
            _validate_sampleable_numeric(
                generator_mapping["aspect_ratio"],
                f"{generator_context}.aspect_ratio",
            )
            _validate_sampleable_vector(
                generator_mapping["center"],
                f"{generator_context}.center",
                gdim=gdim,
            )
        return

    if expr_type == "gaussian_wave_packet":
        if set(params) != {"amplitude", "sigma", "center", "wavevector", "phase"}:
            raise ValueError(
                f"{context}.params must contain exactly ['amplitude', 'center', "
                f"'phase', 'sigma', 'wavevector'] for type '{expr_type}'. Got "
                f"{sorted(params)}."
            )
        _validate_sampleable_numeric(
            params["amplitude"],
            f"{context}.params.amplitude",
        )
        _validate_sampleable_numeric(
            params["sigma"],
            f"{context}.params.sigma",
        )
        _validate_sampleable_numeric(
            params["phase"],
            f"{context}.params.phase",
        )
        _validate_sampleable_vector(
            params["center"],
            f"{context}.params.center",
            gdim=gdim,
        )
        _validate_sampleable_vector(
            params["wavevector"],
            f"{context}.params.wavevector",
            gdim=gdim,
        )
        return

    if expr_type == "sine_waves":
        if set(params) != {"background", "modes"}:
            raise ValueError(
                f"{context}.params must contain exactly ['background', 'modes'] for "
                f"type '{expr_type}'. Got {sorted(params)}."
            )
        _validate_sampleable_numeric(
            params["background"], f"{context}.params.background"
        )
        modes = params["modes"]
        if not isinstance(modes, list) or not modes:
            raise ValueError(f"{context}.params.modes must be a non-empty list.")
        for index, mode in enumerate(modes):
            mode_context = f"{context}.params.modes[{index}]"
            mode_mapping = _as_mapping(mode, mode_context)
            expected_keys = {"amplitude", "cycles", "phase", "angle"}
            if set(mode_mapping) != expected_keys:
                raise ValueError(
                    f"{mode_context} must contain exactly ['amplitude', 'cycles', "
                    f"'phase', 'angle']. Got {sorted(mode_mapping)}."
                )
            _validate_sampleable_numeric(
                mode_mapping["amplitude"], f"{mode_context}.amplitude"
            )
            cycles = mode_mapping["cycles"]
            if not isinstance(cycles, list) or len(cycles) != gdim:
                raise ValueError(
                    f"{mode_context}.cycles must have {gdim} entries in {gdim}D. "
                    f"Got {cycles!r}."
                )
            for axis, cycle_value in enumerate(cycles):
                _validate_sampleable_integer(
                    cycle_value,
                    f"{mode_context}.cycles[{axis}]",
                )
            _validate_sampleable_numeric(mode_mapping["phase"], f"{mode_context}.phase")
            _validate_sampleable_numeric(mode_mapping["angle"], f"{mode_context}.angle")
        return

    if expr_type == "quadrants":
        if set(params) != {"split", "region_values"}:
            raise ValueError(
                f"{context}.params must contain exactly ['region_values', 'split'] "
                f"for type '{expr_type}'. Got {sorted(params)}."
            )
        _validate_sampleable_vector(
            params["split"],
            f"{context}.params.split",
            gdim=gdim,
        )
        region_values = _as_mapping(
            params["region_values"],
            f"{context}.params.region_values",
        )
        expected_region_keys = {format(index, f"0{gdim}b") for index in range(2**gdim)}
        if set(region_values) != expected_region_keys:
            raise ValueError(
                f"{context}.params.region_values must contain exactly "
                f"{sorted(expected_region_keys)}. Got {sorted(region_values)}."
            )
        for key, value in region_values.items():
            _validate_sampleable_numeric(
                value,
                f"{context}.params.region_values.{key}",
            )
        return

    raise ValueError(
        f"{context} uses unsupported initial_condition type '{expr_type}'."
    )


def _validate_initial_condition_expression(
    expr,
    context: str,
    *,
    gdim: int,
) -> None:
    if expr.is_componentwise:
        for label, component in expr.components.items():
            _validate_initial_condition_scalar_expression(
                component,
                f"{context}.components.{label}",
                gdim=gdim,
            )
        return
    _validate_initial_condition_scalar_expression(expr, context, gdim=gdim)


def _infer_domain_dimension(domain_type: str, params: dict[str, Any]) -> int:
    """Infer the spatial dimension from the configured domain."""
    builtin_dims = {
        "interval": 1,
        "rectangle": 2,
        "box": 3,
        "annulus": 2,
        "disk": 2,
        "dumbbell": 2,
        "parallelogram": 2,
        "channel_obstacle": 2,
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


def validate_domain_params(domain_type: str, params: dict[str, Any]) -> None:
    """Validate built-in domain parameters before mesh creation."""

    def _require_keys(*required: str) -> None:
        missing = [name for name in required if name not in params]
        if missing:
            raise ValueError(
                f"{domain_type.capitalize()} domain requires parameters "
                f"{sorted(required)}. Missing {sorted(missing)}."
            )

    def _float_param(name: str, *, positive: bool = False) -> float:
        value = float(params[name])
        if not math.isfinite(value):
            raise ValueError(
                f"{domain_type.capitalize()} domain parameter '{name}' must be "
                f"finite. Got {value}."
            )
        if positive and value <= 0.0:
            raise ValueError(
                f"{domain_type.capitalize()} domain parameter '{name}' must be > 0. "
                f"Got {value}."
            )
        return value

    def _vector_param(name: str, *, length: int) -> list[float]:
        raw = params[name]
        if not isinstance(raw, list) or len(raw) != length:
            raise ValueError(
                f"{domain_type.capitalize()} domain parameter '{name}' must be a "
                f"list with {length} entries. Got {raw!r}."
            )
        values = [float(value) for value in raw]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"{domain_type.capitalize()} domain parameter '{name}' must contain "
                f"only finite numbers. Got {raw!r}."
            )
        return values

    def _positive_int_vector(name: str, *, length: int) -> list[int]:
        raw = params[name]
        if not isinstance(raw, list) or len(raw) != length:
            raise ValueError(
                f"{domain_type.capitalize()} domain parameter '{name}' must be a "
                f"list with {length} entries. Got {raw!r}."
            )
        values: list[int] = []
        for value in raw:
            int_value = int(value)
            if float(int_value) != float(value) or int_value <= 0:
                raise ValueError(
                    f"{domain_type.capitalize()} domain parameter '{name}' must "
                    f"contain positive integers. Got {raw!r}."
                )
            values.append(int_value)
        return values

    if domain_type == "annulus":
        _require_keys("inner_radius", "outer_radius", "mesh_size")
        inner_radius = _float_param("inner_radius", positive=True)
        outer_radius = _float_param("outer_radius", positive=True)
        _float_param("mesh_size", positive=True)
        if inner_radius >= outer_radius:
            raise ValueError(
                "Annulus domain requires 'inner_radius' < 'outer_radius'. "
                f"Got inner_radius={inner_radius} and outer_radius={outer_radius}."
            )
        return

    if domain_type == "disk":
        _require_keys("center", "radius", "mesh_size")
        _vector_param("center", length=2)
        _float_param("radius", positive=True)
        _float_param("mesh_size", positive=True)
        return

    if domain_type == "dumbbell":
        _require_keys(
            "left_center",
            "right_center",
            "lobe_radius",
            "neck_width",
            "mesh_size",
        )
        left_center = _vector_param("left_center", length=2)
        right_center = _vector_param("right_center", length=2)
        lobe_radius = _float_param("lobe_radius", positive=True)
        neck_width = _float_param("neck_width", positive=True)
        _float_param("mesh_size", positive=True)
        if not math.isclose(left_center[1], right_center[1], abs_tol=1.0e-12):
            raise ValueError(
                "Dumbbell domain requires 'left_center' and 'right_center' to "
                "share the same y-coordinate."
            )
        if right_center[0] <= left_center[0]:
            raise ValueError(
                "Dumbbell domain requires 'right_center[0]' to be greater than "
                "'left_center[0]'."
            )
        if neck_width > 2.0 * lobe_radius:
            raise ValueError(
                "Dumbbell domain requires 'neck_width' <= 2 * 'lobe_radius'. "
                f"Got neck_width={neck_width} and lobe_radius={lobe_radius}."
            )
        return

    if domain_type == "parallelogram":
        _require_keys("origin", "axis_x", "axis_y", "mesh_resolution")
        _vector_param("origin", length=2)
        axis_x = _vector_param("axis_x", length=2)
        axis_y = _vector_param("axis_y", length=2)
        _positive_int_vector("mesh_resolution", length=2)
        signed_area = axis_x[0] * axis_y[1] - axis_x[1] * axis_y[0]
        if abs(signed_area) <= 1.0e-12:
            raise ValueError(
                "Parallelogram domain requires linearly independent 'axis_x' and "
                "'axis_y' vectors."
            )
        return

    if domain_type == "channel_obstacle":
        _require_keys(
            "length",
            "height",
            "obstacle_center",
            "obstacle_radius",
            "mesh_size",
        )
        length = _float_param("length", positive=True)
        height = _float_param("height", positive=True)
        obstacle_center = _vector_param("obstacle_center", length=2)
        obstacle_radius = _float_param("obstacle_radius", positive=True)
        _float_param("mesh_size", positive=True)
        cx, cy = obstacle_center
        if cx - obstacle_radius <= 0.0 or cx + obstacle_radius >= length:
            raise ValueError(
                "Channel_obstacle domain requires the circular obstacle to lie "
                "strictly inside the channel in x."
            )
        if cy - obstacle_radius <= 0.0 or cy + obstacle_radius >= height:
            raise ValueError(
                "Channel_obstacle domain requires the circular obstacle to lie "
                "strictly inside the channel in y."
            )


def _load_fragment_catalog() -> dict[str, Any]:
    """Load the shared config-fragment catalog."""
    with open(_FRAGMENT_CATALOG_PATH) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return {}
    return _as_mapping(raw, f"fragment catalog '{_FRAGMENT_CATALOG_PATH}'")


def _lookup_fragment(catalog: dict[str, Any], ref: str, context: str) -> Any:
    """Resolve one dot-separated fragment reference from the shared catalog."""
    target: Any = catalog
    for part in ref.split("."):
        if not isinstance(target, dict) or part not in target:
            raise ValueError(f"{context} references unknown fragment '{ref}'.")
        target = target[part]
    return deepcopy(target)


def _merge_mappings(
    base: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge config mappings with override values winning."""
    merged = {key: deepcopy(value) for key, value in base.items()}
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_mappings(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_refs(
    raw: Any,
    catalog: dict[str, Any],
    *,
    context: str,
    ref_stack: tuple[str, ...] = (),
) -> Any:
    """Expand $ref nodes using the shared fragment catalog."""
    if isinstance(raw, dict):
        if _REF_KEY in raw:
            ref = raw[_REF_KEY]
            if not isinstance(ref, str) or not ref:
                raise ValueError(f"{context}.{_REF_KEY} must be a non-empty string.")
            if ref in ref_stack:
                cycle = " -> ".join((*ref_stack, ref))
                raise ValueError(f"Detected fragment reference cycle: {cycle}.")

            resolved_target = _resolve_refs(
                _lookup_fragment(catalog, ref, context),
                catalog,
                context=f"{context}.{_REF_KEY}",
                ref_stack=(*ref_stack, ref),
            )
            if set(raw) == {_REF_KEY}:
                return resolved_target
            if not isinstance(resolved_target, dict):
                raise ValueError(
                    f"{context} cannot apply local overrides to fragment '{ref}' "
                    "because it does not resolve to a mapping."
                )

            local_overrides = {
                key: _resolve_refs(
                    value,
                    catalog,
                    context=f"{context}.{key}",
                    ref_stack=ref_stack,
                )
                for key, value in raw.items()
                if key != _REF_KEY
            }
            return _merge_mappings(resolved_target, local_overrides)

        return {
            key: _resolve_refs(
                value,
                catalog,
                context=f"{context}.{key}",
                ref_stack=ref_stack,
            )
            for key, value in raw.items()
        }

    if isinstance(raw, list):
        return [
            _resolve_refs(
                value,
                catalog,
                context=f"{context}[{index}]",
                ref_stack=ref_stack,
            )
            for index, value in enumerate(raw)
        ]

    return deepcopy(raw)


def _expand_config_refs(raw: Any) -> dict[str, Any]:
    """Expand shared fragment references in a config file."""
    catalog = _load_fragment_catalog()
    expanded = _resolve_refs(raw, catalog, context="config")
    return _as_mapping(expanded, "config")


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

    def __post_init__(self) -> None:
        validate_domain_params(self.type, self.params)

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
    """PETSc solver strategy and explicit serial / MPI option profiles."""

    strategy: str
    serial: dict[str, str]
    mpi: dict[str, str]

    def options_for_size(self, comm_size: int) -> dict[str, str]:
        """Return the active PETSc options for the communicator size."""
        if comm_size > 1:
            return self.mpi
        return self.serial

    def profile_name_for_size(self, comm_size: int) -> str:
        """Return the active solver-profile name for the communicator size."""
        if comm_size > 1:
            return "mpi"
        return "serial"

    @property
    def options(self) -> dict[str, str]:
        """Return the active PETSc options for the current communicator."""
        return self.options_for_size(MPI.COMM_WORLD.size)

    @property
    def profile_name(self) -> str:
        """Return the active profile name for the current communicator."""
        return self.profile_name_for_size(MPI.COMM_WORLD.size)


@dataclass
class TimeConfig:
    """Time-stepping configuration."""

    dt: float
    t_end: float


@dataclass
class StateStochasticConfig:
    """Dynamic stochastic forcing for one state variable."""

    coupling: str
    intensity: float
    offset: float | None = None


@dataclass
class CoefficientSmoothingConfig:
    """Optional diffusion-style smoothing for static random media."""

    pseudo_dt: float
    steps: int


@dataclass
class CoefficientStochasticConfig:
    """Static stochastic overlay for one scalar coefficient."""

    mode: str
    std: float
    smoothing: CoefficientSmoothingConfig | None = None
    clamp_min: float | None = None


@dataclass
class StochasticConfig:
    """Validated stochastic runtime configuration."""

    states: dict[str, StateStochasticConfig] = field(default_factory=dict)
    coefficients: dict[str, CoefficientStochasticConfig] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        """Return whether any stochastic feature is active."""
        return bool(self.states or self.coefficients)


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
    stochastic: StochasticConfig = field(default_factory=StochasticConfig)

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

    def stochastic_state(self, name: str) -> StateStochasticConfig | None:
        """Return the stochastic forcing configured for one state, if any."""
        return self.stochastic.states.get(name)

    def stochastic_coefficient(self, name: str) -> CoefficientStochasticConfig | None:
        """Return the stochastic overlay configured for one coefficient, if any."""
        return self.stochastic.coefficients.get(name)

    @property
    def has_stochastic(self) -> bool:
        """Return whether stochastic forcing or random media are enabled."""
        return self.stochastic.enabled

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


def _parse_state_stochastic(
    raw: Any,
    context: str,
    *,
    parameters: dict[str, float],
    allowed_couplings: tuple[str, ...],
) -> StateStochasticConfig:
    """Parse one state-level stochastic forcing block."""
    mapping = _as_mapping(raw, context)
    unexpected_keys = set(mapping) - {"coupling", "intensity", "offset"}
    if unexpected_keys:
        raise ValueError(f"{context} has unsupported keys {sorted(unexpected_keys)}.")

    coupling = str(_require(mapping, "coupling", context))
    if coupling not in _VALID_STOCHASTIC_STATE_COUPLINGS:
        raise ValueError(
            f"{context}.coupling must be one of "
            f"{sorted(_VALID_STOCHASTIC_STATE_COUPLINGS)}. Got '{coupling}'."
        )
    if coupling not in allowed_couplings:
        raise ValueError(
            f"{context}.coupling '{coupling}' is not supported by this state. "
            f"Allowed couplings: {sorted(allowed_couplings)}."
        )

    intensity = _resolve_numeric_literal_or_param_ref(
        _require(mapping, "intensity", context),
        parameters,
        f"{context}.intensity",
    )
    if intensity < 0.0:
        raise ValueError(f"{context}.intensity must be non-negative.")

    raw_offset = mapping.get("offset")
    if coupling == "saturating_self":
        if raw_offset is None:
            raise ValueError(
                f"{context} with coupling 'saturating_self' requires 'offset'."
            )
        offset = _resolve_numeric_literal_or_param_ref(
            raw_offset,
            parameters,
            f"{context}.offset",
        )
        if offset <= 0.0:
            raise ValueError(f"{context}.offset must be positive.")
    else:
        if raw_offset is not None:
            raise ValueError(
                f"{context}.offset is only valid when coupling is 'saturating_self'."
            )
        offset = None

    return StateStochasticConfig(
        coupling=coupling,
        intensity=intensity,
        offset=offset,
    )


def _parse_coefficient_stochastic(
    raw: Any,
    context: str,
    *,
    parameters: dict[str, float],
) -> CoefficientStochasticConfig:
    """Parse one coefficient-level stochastic randomization block."""
    mapping = _as_mapping(raw, context)
    unexpected_keys = set(mapping) - {"mode", "std", "smoothing", "clamp_min"}
    if unexpected_keys:
        raise ValueError(f"{context} has unsupported keys {sorted(unexpected_keys)}.")

    mode = str(_require(mapping, "mode", context))
    if mode not in _VALID_STOCHASTIC_COEFFICIENT_MODES:
        raise ValueError(
            f"{context}.mode must be one of "
            f"{sorted(_VALID_STOCHASTIC_COEFFICIENT_MODES)}. Got '{mode}'."
        )

    std = _resolve_numeric_literal_or_param_ref(
        _require(mapping, "std", context),
        parameters,
        f"{context}.std",
    )
    if std < 0.0:
        raise ValueError(f"{context}.std must be non-negative.")

    clamp_min = mapping.get("clamp_min")
    if clamp_min is not None:
        clamp_min = _resolve_numeric_literal_or_param_ref(
            clamp_min,
            parameters,
            f"{context}.clamp_min",
        )

    smoothing_raw = mapping.get("smoothing")
    if smoothing_raw is None:
        smoothing = None
    else:
        smoothing_context = f"{context}.smoothing"
        smoothing_mapping = _as_mapping(smoothing_raw, smoothing_context)
        if set(smoothing_mapping) != {"pseudo_dt", "steps"}:
            raise ValueError(
                f"{smoothing_context} must contain exactly ['pseudo_dt', 'steps']. "
                f"Got {sorted(smoothing_mapping)}."
            )
        pseudo_dt = _resolve_numeric_literal_or_param_ref(
            smoothing_mapping["pseudo_dt"],
            parameters,
            f"{smoothing_context}.pseudo_dt",
        )
        if pseudo_dt <= 0.0:
            raise ValueError(f"{smoothing_context}.pseudo_dt must be positive.")
        steps = int(smoothing_mapping["steps"])
        if steps <= 0:
            raise ValueError(f"{smoothing_context}.steps must be positive.")
        smoothing = CoefficientSmoothingConfig(
            pseudo_dt=pseudo_dt,
            steps=steps,
        )

    return CoefficientStochasticConfig(
        mode=mode,
        std=std,
        smoothing=smoothing,
        clamp_min=clamp_min,
    )


def _parse_stochastic(
    raw: Any,
    *,
    spec,
    parameters: dict[str, float],
) -> StochasticConfig:
    """Parse the top-level stochastic configuration block."""
    if raw is None:
        return StochasticConfig()

    mapping = _as_mapping(raw, "stochastic")
    unexpected_keys = set(mapping) - {"states", "coefficients"}
    if unexpected_keys:
        raise ValueError(f"stochastic has unsupported keys {sorted(unexpected_keys)}.")

    states_raw = _as_mapping(mapping.get("states", {}), "stochastic.states")
    coefficients_raw = _as_mapping(
        mapping.get("coefficients", {}),
        "stochastic.coefficients",
    )

    state_configs: dict[str, StateStochasticConfig] = {}
    for state_name, state_raw in states_raw.items():
        if state_name not in spec.states:
            raise ValueError(
                f"stochastic.states references unknown state '{state_name}'. "
                f"Available states: {sorted(spec.states)}."
            )
        state_spec = spec.states[state_name]
        if not state_spec.stochastic_couplings:
            raise ValueError(
                f"Preset '{spec.name}' state '{state_name}' does not support "
                "stochastic forcing."
            )
        state_configs[state_name] = _parse_state_stochastic(
            state_raw,
            f"stochastic.states.{state_name}",
            parameters=parameters,
            allowed_couplings=state_spec.stochastic_couplings,
        )

    coefficient_configs: dict[str, CoefficientStochasticConfig] = {}
    for coefficient_name, coefficient_raw in coefficients_raw.items():
        if coefficient_name not in spec.coefficients:
            raise ValueError(
                f"stochastic.coefficients references unknown coefficient "
                f"'{coefficient_name}'. Available coefficients: "
                f"{sorted(spec.coefficients)}."
            )
        coefficient_spec = spec.coefficients[coefficient_name]
        if not coefficient_spec.allow_randomization:
            raise ValueError(
                f"Preset '{spec.name}' coefficient '{coefficient_name}' does not "
                "support stochastic randomization."
            )
        coefficient_configs[coefficient_name] = _parse_coefficient_stochastic(
            coefficient_raw,
            f"stochastic.coefficients.{coefficient_name}",
            parameters=parameters,
        )

    return StochasticConfig(
        states=state_configs,
        coefficients=coefficient_configs,
    )


def load_config(path: str | Path) -> SimulationConfig:
    """Load and validate a simulation config from YAML."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    raw = _expand_config_refs(raw)
    unexpected_top_level_keys = set(raw) - _TOP_LEVEL_CONFIG_KEYS
    if unexpected_top_level_keys:
        raise ValueError(
            f"config has unsupported top-level keys "
            f"{sorted(unexpected_top_level_keys)}."
        )

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

    stochastic = _parse_stochastic(
        raw.get("stochastic"),
        spec=spec,
        parameters=parameters,
    )

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
    strategy = str(_require(solver_raw, "strategy", "solver"))
    if strategy not in ALL_SOLVER_STRATEGIES:
        raise ValueError(
            f"solver.strategy must be one of {sorted(ALL_SOLVER_STRATEGIES)}. "
            f"Got '{strategy}'."
        )
    unexpected_solver_keys = set(solver_raw) - {"strategy", "serial", "mpi"}
    if unexpected_solver_keys:
        raise ValueError(
            f"solver has unsupported keys {sorted(unexpected_solver_keys)}."
        )
    solver = SolverConfig(
        strategy=strategy,
        serial={
            str(k): str(v)
            for k, v in _as_mapping(
                _require(solver_raw, "serial", "solver"),
                "solver.serial",
            ).items()
        },
        mpi={
            str(k): str(v)
            for k, v in _as_mapping(
                _require(solver_raw, "mpi", "solver"),
                "solver.mpi",
            ).items()
        },
    )

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
            _validate_initial_condition_expression(
                initial_condition,
                f"{context}.initial_condition",
                gdim=gdim,
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
        stochastic=stochastic,
    )

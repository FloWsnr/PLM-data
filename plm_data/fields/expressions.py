"""Field-expression configuration helpers."""

from typing import Any

from plm_data.core.runtime_config import FieldExpressionConfig

_COMPONENT_LABELS = ("x", "y", "z")


def resolve_param_ref(value: Any, parameters: dict[str, float]) -> float:
    """Resolve a literal number or ``param:<name>`` reference."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.startswith("param:"):
        name = value[len("param:") :]
        if name not in parameters:
            raise ValueError(
                f"Parameter reference '{value}' not found. "
                f"Available parameters: {list(parameters.keys())}"
            )
        return float(parameters[name])
    raise ValueError(
        f"Cannot resolve value '{value}'. "
        f"Expected a number or 'param:<name>' reference."
    )


def normalize_field_config(value: Any) -> dict:
    """Normalize a field value to the legacy ``{type, params}`` shape."""
    if isinstance(value, (int, float)):
        return {"type": "constant", "params": {"value": value}}
    if isinstance(value, str) and value.startswith("param:"):
        return {"type": "constant", "params": {"value": value}}
    if isinstance(value, dict):
        if "type" not in value:
            raise ValueError(f"Field config dict must have a 'type' key. Got: {value}")
        return value
    raise ValueError(
        f"Invalid field config: {value!r}. "
        f"Expected a number, 'param:<name>' string, or {{type, params}} dict."
    )


def component_labels_for_dim(gdim: int) -> tuple[str, ...]:
    """Return active vector component labels for the dimension."""
    return _COMPONENT_LABELS[:gdim]


def scalar_expression_to_config(expr: FieldExpressionConfig) -> dict:
    """Convert a scalar field expression config to ``{type, params}`` form."""
    if expr.is_componentwise:
        raise ValueError(
            "Expected a scalar field expression, got component-wise config"
        )
    if expr.type is None:
        raise ValueError("Scalar field expression must define a 'type'")
    return {"type": expr.type, "params": expr.params}


def component_expressions(
    expr: FieldExpressionConfig,
    gdim: int,
) -> dict[str, FieldExpressionConfig]:
    """Expand a vector field expression into scalar component expressions."""
    labels = component_labels_for_dim(gdim)
    if expr.components:
        if set(expr.components) != set(labels):
            raise ValueError(
                f"Vector field components must match {list(labels)} in {gdim}D. "
                f"Got {sorted(expr.components)}."
            )
        return {label: expr.components[label] for label in labels}

    if expr.type in {"none", "zero", "custom"}:
        return {
            label: FieldExpressionConfig(type=expr.type, params=dict(expr.params))
            for label in labels
        }

    raise ValueError(
        "Vector field expressions must use explicit components or a field-level "
        "'none', 'zero', or 'custom' type"
    )


def require_field_param(params: dict, key: str, field_type: str):
    """Require a field parameter, raising a clear error if missing."""
    if key not in params:
        raise ValueError(
            f"Missing required parameter '{key}' for field type '{field_type}'. "
            f"Got params: {params}"
        )
    return params[key]


def resolve_sine_waves_mode(
    mode: dict[str, Any],
    parameters: dict[str, float],
    gdim: int,
) -> tuple[float, list[float], float, float]:
    """Resolve one sine-waves mode into numeric amplitude/cycles/phase/angle."""
    if not isinstance(mode, dict):
        raise ValueError(
            "sine_waves modes must be mappings with amplitude, cycles, and phase keys."
        )

    amplitude = resolve_param_ref(
        require_field_param(mode, "amplitude", "sine_waves"),
        parameters,
    )
    cycles_raw = require_field_param(mode, "cycles", "sine_waves")
    phase = resolve_param_ref(
        require_field_param(mode, "phase", "sine_waves"), parameters
    )
    angle = resolve_param_ref(mode.get("angle", 0.0), parameters)

    if not isinstance(cycles_raw, list) or len(cycles_raw) != gdim:
        raise ValueError(
            f"sine_waves cycles must have {gdim} entries in {gdim}D. "
            f"Got {len(cycles_raw) if isinstance(cycles_raw, list) else 'non-list'}."
        )

    return (
        amplitude,
        [resolve_param_ref(cycle, parameters) for cycle in cycles_raw],
        phase,
        angle,
    )


def sine_waves_dimension(modes: list[Any]) -> int:
    """Return the active dimension implied by a sine-waves modes list."""
    first_mode = modes[0]
    if not isinstance(first_mode, dict):
        raise ValueError(
            "sine_waves modes must be mappings with amplitude, cycles, and phase keys."
        )
    cycles = require_field_param(first_mode, "cycles", "sine_waves")
    if not isinstance(cycles, list) or not cycles:
        raise ValueError(
            "sine_waves modes must provide a non-empty 'cycles' list with one entry "
            "per active axis."
        )
    return len(cycles)


def is_exact_zero_field_expression(
    expr: FieldExpressionConfig,
    parameters: dict[str, float],
) -> bool:
    """Return whether a field config is exactly zero after parameter resolution."""
    if expr.is_componentwise:
        return all(
            is_exact_zero_field_expression(component, parameters)
            for component in expr.components.values()
        )

    if expr.type in {"none", "zero"}:
        return True

    if expr.type == "constant":
        return (
            resolve_param_ref(
                require_field_param(expr.params, "value", expr.type), parameters
            )
            == 0.0
        )

    return False

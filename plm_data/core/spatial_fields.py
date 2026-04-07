"""Shared scalar and vector spatial field helpers."""

from typing import Any, Callable

import numpy as np
import ufl
from dolfinx import default_real_type, fem
from dolfinx import mesh as dmesh

from plm_data.core.config import FieldExpressionConfig

_COMPONENT_LABELS = ("x", "y", "z")
_PI = np.pi


def resolve_param_ref(value: Any, parameters: dict[str, float]) -> float:
    """Resolve a value that may be a 'param:name' reference.

    Returns the float directly if already numeric, or looks up the
    referenced parameter name.
    """
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
    """Normalize a field value (float, string, or dict) to {type, params} form.

    - float/int → {type: "constant", params: {value: <float>}}
    - "param:name" → {type: "constant", params: {value: "param:name"}}
    - dict with type/params → returned as-is
    """
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
    """Convert a scalar field expression config to the old {type, params} shape."""
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


def _require_param(params: dict, key: str, field_type: str):
    """Require a parameter, raising a clear error if missing."""
    if key not in params:
        raise ValueError(
            f"Missing required parameter '{key}' for field type '{field_type}'. "
            f"Got params: {params}"
        )
    return params[key]


def _resolve_sine_waves_mode(
    mode: dict[str, Any],
    parameters: dict[str, float],
    gdim: int,
) -> tuple[float, list[float], float, float]:
    if not isinstance(mode, dict):
        raise ValueError(
            "sine_waves modes must be mappings with amplitude, cycles, and phase keys."
        )

    amplitude = resolve_param_ref(
        _require_param(mode, "amplitude", "sine_waves"),
        parameters,
    )
    cycles_raw = _require_param(mode, "cycles", "sine_waves")
    phase = resolve_param_ref(_require_param(mode, "phase", "sine_waves"), parameters)
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


def _sine_waves_dimension(modes: list[Any]) -> int:
    first_mode = modes[0]
    if not isinstance(first_mode, dict):
        raise ValueError(
            "sine_waves modes must be mappings with amplitude, cycles, and phase keys."
        )
    cycles = _require_param(first_mode, "cycles", "sine_waves")
    if not isinstance(cycles, list) or not cycles:
        raise ValueError(
            "sine_waves modes must provide a non-empty 'cycles' list with one entry "
            "per active axis."
        )
    return len(cycles)


def _rotated_coordinates_ufl(x, angle: float, gdim: int) -> list:
    if angle == 0.0 or gdim < 2:
        return [x[axis] for axis in range(gdim)]

    cos_angle = float(np.cos(angle))
    sin_angle = float(np.sin(angle))
    rotated = [
        cos_angle * x[0] - sin_angle * x[1],
        sin_angle * x[0] + cos_angle * x[1],
    ]
    rotated.extend(x[axis] for axis in range(2, gdim))
    return rotated


def _rotated_coordinates_numpy(x: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0.0 or x.shape[0] < 2:
        return x

    rotated = np.array(x, copy=True)
    cos_angle = float(np.cos(angle))
    sin_angle = float(np.sin(angle))
    rotated[0] = cos_angle * x[0] - sin_angle * x[1]
    rotated[1] = sin_angle * x[0] + cos_angle * x[1]
    return rotated


def _build_sine_waves_expression(
    x,
    params: dict,
    parameters: dict[str, float],
    gdim: int,
):
    """Build a sine_waves UFL expression."""
    background = resolve_param_ref(
        _require_param(params, "background", "sine_waves"), parameters
    )
    modes = _require_param(params, "modes", "sine_waves")

    if not isinstance(modes, list) or not modes:
        raise ValueError("sine_waves requires a non-empty 'modes' list.")

    expr = ufl.as_ufl(background)

    for mode in modes:
        amplitude, cycles, phase, angle = _resolve_sine_waves_mode(
            mode,
            parameters,
            gdim,
        )
        coordinates = _rotated_coordinates_ufl(x, angle, gdim)
        mode_expr = ufl.as_ufl(amplitude)
        for axis, cycle in enumerate(cycles):
            if cycle == 0.0:
                continue
            mode_expr = mode_expr * ufl.sin(_PI * cycle * coordinates[axis] + phase)

        expr = expr + mode_expr

    return expr


def _build_affine_expression(
    x,
    params: dict,
    parameters: dict[str, float],
):
    expr = ufl.as_ufl(resolve_param_ref(params.get("constant", 0.0), parameters))
    axis_keys = {"x": 0, "y": 1, "z": 2}
    for key, axis in axis_keys.items():
        if key not in params:
            continue
        expr = expr + resolve_param_ref(params[key], parameters) * x[axis]
    return expr


def _build_affine_interpolator(
    params: dict,
    parameters: dict[str, float],
) -> Callable[[np.ndarray], np.ndarray]:
    constant = resolve_param_ref(params.get("constant", 0.0), parameters)
    axis_coefficients = {
        axis: resolve_param_ref(params[label], parameters)
        for label, axis in {"x": 0, "y": 1, "z": 2}.items()
        if label in params
    }

    def _affine(x, c=constant, coeffs=axis_coefficients):
        result = np.full(x.shape[1], c)
        for axis, coefficient in coeffs.items():
            result = result + coefficient * x[axis]
        return result

    return _affine


# ---------------------------------------------------------------------------
# UFL renderer — produces symbolic expressions for variational forms
# ---------------------------------------------------------------------------


def build_ufl_field(
    msh: dmesh.Mesh,
    field_config: dict,
    parameters: dict[str, float],
) -> "ufl.core.expr.Expr | None":  # type: ignore[reportAttributeAccessIssue]
    """Build a UFL expression from a normalized field config.

    Args:
        msh: The DOLFINx mesh (needed for spatial coordinates).
        field_config: A dict with 'type' and 'params' keys.
        parameters: PDE parameters dict for resolving 'param:name' refs.

    Returns:
        A UFL expression that can be used in variational forms.
    """
    field_type = field_config["type"]
    p = field_config.get("params", {})
    x = ufl.SpatialCoordinate(msh)

    if field_type in ("none", "zero"):
        return fem.Constant(msh, default_real_type(0.0))

    elif field_type == "constant":
        value = resolve_param_ref(_require_param(p, "value", field_type), parameters)
        return fem.Constant(msh, default_real_type(value))

    elif field_type == "gaussian_bump":
        amplitude = resolve_param_ref(
            _require_param(p, "amplitude", field_type), parameters
        )
        sigma = resolve_param_ref(_require_param(p, "sigma", field_type), parameters)
        center = _require_param(p, "center", field_type)
        gdim = msh.geometry.dim
        if len(center) != gdim:
            raise ValueError(
                f"gaussian_bump center has {len(center)} components but mesh is {gdim}D"
            )
        r_sq = sum((x[i] - center[i]) ** 2 for i in range(gdim))
        return amplitude * ufl.exp(-r_sq / (2 * sigma**2))

    elif field_type == "radial_cosine":
        base = resolve_param_ref(_require_param(p, "base", field_type), parameters)
        amplitude = resolve_param_ref(
            _require_param(p, "amplitude", field_type), parameters
        )
        frequency = resolve_param_ref(
            _require_param(p, "frequency", field_type), parameters
        )
        center = _require_param(p, "center", field_type)
        gdim = msh.geometry.dim
        if len(center) != gdim:
            raise ValueError(
                f"radial_cosine center has {len(center)} components but mesh is {gdim}D"
            )
        r = ufl.sqrt(sum((x[i] - center[i]) ** 2 for i in range(gdim)))
        return ufl.as_ufl(base) + amplitude * ufl.cos(frequency * r)

    elif field_type == "affine":
        return _build_affine_expression(x, p, parameters)

    elif field_type == "step":
        value_left = resolve_param_ref(
            _require_param(p, "value_left", field_type), parameters
        )
        value_right = resolve_param_ref(
            _require_param(p, "value_right", field_type), parameters
        )
        x_split = resolve_param_ref(
            _require_param(p, "x_split", field_type), parameters
        )
        axis = int(_require_param(p, "axis", field_type))
        return ufl.conditional(ufl.lt(x[axis], x_split), value_left, value_right)

    elif field_type == "sine_waves":
        return _build_sine_waves_expression(x, p, parameters, msh.geometry.dim)

    elif field_type == "custom":
        return None

    else:
        raise ValueError(f"Unknown field type: '{field_type}'")


def build_vector_ufl_field(
    msh: dmesh.Mesh,
    expr: FieldExpressionConfig,
    parameters: dict[str, float],
) -> "ufl.core.expr.Expr | None":  # type: ignore[reportAttributeAccessIssue]
    """Build a vector-valued UFL expression from a field expression config."""
    gdim = msh.geometry.dim
    if not expr.is_componentwise:
        if expr.type in {"none", "zero"}:
            return fem.Constant(msh, np.zeros(gdim, dtype=default_real_type))
        if expr.type == "custom":
            return None

    components = component_expressions(expr, gdim)
    component_exprs = []
    for label in component_labels_for_dim(gdim):
        scalar_expr = build_ufl_field(
            msh,
            scalar_expression_to_config(components[label]),
            parameters,
        )
        if scalar_expr is None:
            raise ValueError(
                f"Component '{label}' cannot use 'custom' inside a vector expression"
            )
        component_exprs.append(scalar_expr)
    return ufl.as_vector(component_exprs)


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
                _require_param(expr.params, "value", expr.type), parameters
            )
            == 0.0
        )

    return False


# ---------------------------------------------------------------------------
# Numpy interpolation renderer — produces callables for Function.interpolate()
# ---------------------------------------------------------------------------


def build_interpolator(
    field_config: dict,
    parameters: dict[str, float],
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Build a numpy callable from a normalized field config.

    The callable has signature (x: ndarray) -> ndarray, where x has shape
    (gdim, num_points), matching DOLFINx's interpolation convention.

    Args:
        field_config: A dict with 'type' and 'params' keys.
        parameters: PDE parameters dict for resolving 'param:name' refs.

    Returns:
        A callable for use with Function.interpolate(), or None for 'custom'.
    """
    field_type = field_config["type"]
    p = field_config.get("params", {})

    if field_type in ("none", "zero"):
        return lambda x: np.zeros(x.shape[1])

    elif field_type == "constant":
        value = resolve_param_ref(_require_param(p, "value", field_type), parameters)
        return lambda x, v=value: np.full(x.shape[1], v)

    elif field_type == "gaussian_bump":
        amplitude = resolve_param_ref(
            _require_param(p, "amplitude", field_type), parameters
        )
        sigma = resolve_param_ref(_require_param(p, "sigma", field_type), parameters)
        center = _require_param(p, "center", field_type)

        def _gaussian(x, amp=amplitude, sig=sigma, ctr=center):
            gdim = min(len(ctr), x.shape[0])
            r_sq = sum((x[i] - ctr[i]) ** 2 for i in range(gdim))
            return amp * np.exp(-r_sq / (2 * sig**2))

        return _gaussian

    elif field_type == "radial_cosine":
        base = resolve_param_ref(_require_param(p, "base", field_type), parameters)
        amplitude = resolve_param_ref(
            _require_param(p, "amplitude", field_type), parameters
        )
        frequency = resolve_param_ref(
            _require_param(p, "frequency", field_type), parameters
        )
        center = _require_param(p, "center", field_type)

        def _radial_cosine(
            x, base_value=base, amp=amplitude, freq=frequency, ctr=center
        ):
            gdim = min(len(ctr), x.shape[0])
            r = np.sqrt(sum((x[i] - ctr[i]) ** 2 for i in range(gdim)))
            return base_value + amp * np.cos(freq * r)

        return _radial_cosine

    elif field_type == "affine":
        return _build_affine_interpolator(p, parameters)

    elif field_type == "step":
        value_left = resolve_param_ref(
            _require_param(p, "value_left", field_type), parameters
        )
        value_right = resolve_param_ref(
            _require_param(p, "value_right", field_type), parameters
        )
        x_split = resolve_param_ref(
            _require_param(p, "x_split", field_type), parameters
        )
        axis = int(_require_param(p, "axis", field_type))

        def _step(x, vl=value_left, vr=value_right, xs=x_split, ax=axis):
            return np.where(x[ax] < xs, vl, vr)

        return _step

    elif field_type == "sine_waves":
        return _build_sine_waves_interpolator(p, parameters)

    elif field_type == "custom":
        return None

    else:
        raise ValueError(f"Unknown field type: '{field_type}'")


def _build_sine_waves_interpolator(
    params: dict,
    parameters: dict[str, float],
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a sine_waves interpolator function."""
    background = resolve_param_ref(
        _require_param(params, "background", "sine_waves"), parameters
    )
    modes = _require_param(params, "modes", "sine_waves")

    if not isinstance(modes, list) or not modes:
        raise ValueError("sine_waves requires a non-empty 'modes' list.")
    gdim = _sine_waves_dimension(modes)

    def _sine_waves_interpolator(x: np.ndarray) -> np.ndarray:
        values = np.full(x.shape[1], background, dtype=float)

        for mode in modes:
            amplitude, cycles, phase, angle = _resolve_sine_waves_mode(
                mode,
                parameters,
                gdim,
            )
            coordinates = _rotated_coordinates_numpy(x[:gdim], angle)
            mode_values = np.full(x.shape[1], float(amplitude), dtype=float)
            for axis, cycle in enumerate(cycles):
                if cycle == 0.0:
                    continue
                mode_values *= np.sin(_PI * cycle * coordinates[axis] + phase)
            values = values + mode_values

        return values

    return _sine_waves_interpolator


def build_vector_interpolator(
    expr: FieldExpressionConfig,
    gdim: int,
    parameters: dict[str, float],
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Build a vector-valued interpolator from a field expression config."""
    if not expr.is_componentwise and expr.type == "custom":
        return None

    components = component_expressions(expr, gdim)
    scalar_interps = []
    for label in component_labels_for_dim(gdim):
        scalar_interp = build_interpolator(
            scalar_expression_to_config(components[label]),
            parameters,
        )
        if scalar_interp is None:
            raise ValueError(
                f"Component '{label}' cannot use 'custom' inside a vector expression"
            )
        scalar_interps.append(scalar_interp)

    def _vector_interpolator(x: np.ndarray) -> np.ndarray:
        values = np.zeros((gdim, x.shape[1]))
        for i, interp in enumerate(scalar_interps):
            values[i, :] = interp(x)
        return values

    return _vector_interpolator

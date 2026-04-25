"""Numpy interpolation callables for field expressions."""

from collections.abc import Callable

import numpy as np

from plm_data.core.runtime_config import FieldExpressionConfig
from plm_data.fields.expressions import (
    component_expressions,
    component_labels_for_dim,
    require_field_param,
    resolve_param_ref,
    resolve_sine_waves_mode,
    scalar_expression_to_config,
    sine_waves_dimension,
)

_PI = np.pi


def _rotated_coordinates_numpy(x: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0.0 or x.shape[0] < 2:
        return x

    rotated = np.array(x, copy=True)
    cos_angle = float(np.cos(angle))
    sin_angle = float(np.sin(angle))
    rotated[0] = cos_angle * x[0] - sin_angle * x[1]
    rotated[1] = sin_angle * x[0] + cos_angle * x[1]
    return rotated


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


def build_interpolator(
    field_config: dict,
    parameters: dict[str, float],
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Build a numpy callable from a normalized field config."""
    field_type = field_config["type"]
    p = field_config.get("params", {})

    if field_type in ("none", "zero"):
        return lambda x: np.zeros(x.shape[1])

    if field_type == "constant":
        value = resolve_param_ref(
            require_field_param(p, "value", field_type), parameters
        )
        return lambda x, v=value: np.full(x.shape[1], v)

    if field_type == "gaussian_bump":
        amplitude = resolve_param_ref(
            require_field_param(p, "amplitude", field_type), parameters
        )
        sigma = resolve_param_ref(
            require_field_param(p, "sigma", field_type), parameters
        )
        center = require_field_param(p, "center", field_type)

        def _gaussian(x, amp=amplitude, sig=sigma, ctr=center):
            gdim = min(len(ctr), x.shape[0])
            r_sq = sum((x[i] - ctr[i]) ** 2 for i in range(gdim))
            return amp * np.exp(-r_sq / (2 * sig**2))

        return _gaussian

    if field_type == "radial_cosine":
        base = resolve_param_ref(require_field_param(p, "base", field_type), parameters)
        amplitude = resolve_param_ref(
            require_field_param(p, "amplitude", field_type), parameters
        )
        frequency = resolve_param_ref(
            require_field_param(p, "frequency", field_type), parameters
        )
        center = require_field_param(p, "center", field_type)

        def _radial_cosine(
            x, base_value=base, amp=amplitude, freq=frequency, ctr=center
        ):
            gdim = min(len(ctr), x.shape[0])
            r = np.sqrt(sum((x[i] - ctr[i]) ** 2 for i in range(gdim)))
            return base_value + amp * np.cos(freq * r)

        return _radial_cosine

    if field_type == "affine":
        return _build_affine_interpolator(p, parameters)

    if field_type == "step":
        value_left = resolve_param_ref(
            require_field_param(p, "value_left", field_type), parameters
        )
        value_right = resolve_param_ref(
            require_field_param(p, "value_right", field_type), parameters
        )
        x_split = resolve_param_ref(
            require_field_param(p, "x_split", field_type), parameters
        )
        axis = int(require_field_param(p, "axis", field_type))

        def _step(x, vl=value_left, vr=value_right, xs=x_split, ax=axis):
            return np.where(x[ax] < xs, vl, vr)

        return _step

    if field_type == "sine_waves":
        return _build_sine_waves_interpolator(p, parameters)

    if field_type == "custom":
        return None

    raise ValueError(f"Unknown field type: '{field_type}'")


def _build_sine_waves_interpolator(
    params: dict,
    parameters: dict[str, float],
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a sine_waves interpolator function."""
    background = resolve_param_ref(
        require_field_param(params, "background", "sine_waves"), parameters
    )
    modes = require_field_param(params, "modes", "sine_waves")

    if not isinstance(modes, list) or not modes:
        raise ValueError("sine_waves requires a non-empty 'modes' list.")
    gdim = sine_waves_dimension(modes)

    def _sine_waves_interpolator(x: np.ndarray) -> np.ndarray:
        values = np.full(x.shape[1], background, dtype=float)

        for mode in modes:
            amplitude, cycles, phase, angle = resolve_sine_waves_mode(
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

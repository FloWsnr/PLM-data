"""Shared spatial field type system.

Converts {type, params} field configs into UFL expressions (for variational
forms) or numpy callables (for interpolation into DOLFINx Functions).
Used by source terms, boundary condition values, and initial conditions.
"""

from typing import Any, Callable

import numpy as np
import ufl
from dolfinx import mesh as dmesh


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


def _require_param(params: dict, key: str, field_type: str):
    """Require a parameter, raising a clear error if missing."""
    if key not in params:
        raise ValueError(
            f"Missing required parameter '{key}' for field type '{field_type}'. "
            f"Got params: {params}"
        )
    return params[key]


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
        return ufl.as_ufl(0.0)

    elif field_type == "constant":
        value = resolve_param_ref(_require_param(p, "value", field_type), parameters)
        return ufl.as_ufl(value)

    elif field_type == "sine_product":
        amplitude = resolve_param_ref(
            _require_param(p, "amplitude", field_type), parameters
        )
        expr = ufl.as_ufl(amplitude)
        axis_keys = {"kx": 0, "ky": 1, "kz": 2}
        found_any = False
        for key, axis in axis_keys.items():
            if key in p:
                k = resolve_param_ref(p[key], parameters)
                expr = expr * ufl.sin(k * ufl.pi * x[axis])  # type: ignore[reportOperatorIssue]
                found_any = True
        if not found_any:
            raise ValueError(
                f"sine_product requires at least one of kx, ky, kz. Got params: {p}"
            )
        return expr

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

    elif field_type == "custom":
        return None

    else:
        raise ValueError(f"Unknown field type: '{field_type}'")


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

    elif field_type == "sine_product":
        amplitude = resolve_param_ref(
            _require_param(p, "amplitude", field_type), parameters
        )
        axis_keys = {"kx": 0, "ky": 1, "kz": 2}
        axes = {}
        for key, axis in axis_keys.items():
            if key in p:
                axes[axis] = resolve_param_ref(p[key], parameters)
        if not axes:
            raise ValueError(
                f"sine_product requires at least one of kx, ky, kz. Got params: {p}"
            )

        def _sine_product(x, amp=amplitude, ax=axes):
            result = np.full(x.shape[1], amp)
            for axis, k in ax.items():
                result = result * np.sin(k * np.pi * x[axis])
            return result

        return _sine_product

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

    elif field_type == "custom":
        return None

    else:
        raise ValueError(f"Unknown field type: '{field_type}'")

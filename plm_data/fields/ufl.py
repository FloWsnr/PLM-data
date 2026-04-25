"""UFL rendering for field expressions."""

import numpy as np
import ufl
from dolfinx import default_real_type, fem
from dolfinx import mesh as dmesh

from plm_data.core.runtime_config import FieldExpressionConfig
from plm_data.fields.expressions import (
    component_expressions,
    component_labels_for_dim,
    require_field_param,
    resolve_param_ref,
    resolve_sine_waves_mode,
    scalar_expression_to_config,
)

_PI = np.pi


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


def _build_sine_waves_expression(
    x,
    params: dict,
    parameters: dict[str, float],
    gdim: int,
):
    """Build a sine_waves UFL expression."""
    background = resolve_param_ref(
        require_field_param(params, "background", "sine_waves"), parameters
    )
    modes = require_field_param(params, "modes", "sine_waves")

    if not isinstance(modes, list) or not modes:
        raise ValueError("sine_waves requires a non-empty 'modes' list.")

    expr = ufl.as_ufl(background)

    for mode in modes:
        amplitude, cycles, phase, angle = resolve_sine_waves_mode(
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


def build_ufl_field(
    msh: dmesh.Mesh,
    field_config: dict,
    parameters: dict[str, float],
) -> "ufl.core.expr.Expr | None":  # type: ignore[reportAttributeAccessIssue]
    """Build a UFL expression from a normalized field config."""
    field_type = field_config["type"]
    p = field_config.get("params", {})
    x = ufl.SpatialCoordinate(msh)

    if field_type in ("none", "zero"):
        return fem.Constant(msh, default_real_type(0.0))

    if field_type == "constant":
        value = resolve_param_ref(
            require_field_param(p, "value", field_type), parameters
        )
        return fem.Constant(msh, default_real_type(value))

    if field_type == "gaussian_bump":
        amplitude = resolve_param_ref(
            require_field_param(p, "amplitude", field_type), parameters
        )
        sigma = resolve_param_ref(
            require_field_param(p, "sigma", field_type), parameters
        )
        center = require_field_param(p, "center", field_type)
        gdim = msh.geometry.dim
        if len(center) != gdim:
            raise ValueError(
                f"gaussian_bump center has {len(center)} components but mesh is {gdim}D"
            )
        r_sq = sum((x[i] - center[i]) ** 2 for i in range(gdim))
        return amplitude * ufl.exp(-r_sq / (2 * sigma**2))

    if field_type == "radial_cosine":
        base = resolve_param_ref(require_field_param(p, "base", field_type), parameters)
        amplitude = resolve_param_ref(
            require_field_param(p, "amplitude", field_type), parameters
        )
        frequency = resolve_param_ref(
            require_field_param(p, "frequency", field_type), parameters
        )
        center = require_field_param(p, "center", field_type)
        gdim = msh.geometry.dim
        if len(center) != gdim:
            raise ValueError(
                f"radial_cosine center has {len(center)} components but mesh is {gdim}D"
            )
        r = ufl.sqrt(sum((x[i] - center[i]) ** 2 for i in range(gdim)))
        return ufl.as_ufl(base) + amplitude * ufl.cos(frequency * r)

    if field_type == "affine":
        return _build_affine_expression(x, p, parameters)

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
        return ufl.conditional(ufl.lt(x[axis], x_split), value_left, value_right)

    if field_type == "sine_waves":
        return _build_sine_waves_expression(x, p, parameters, msh.geometry.dim)

    if field_type == "custom":
        return None

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

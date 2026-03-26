"""Shared helpers for Maxwell-family presets."""

from __future__ import annotations

import ufl
from dolfinx import fem

from plm_data.core.boundary_conditions import apply_vector_dirichlet_bcs
from plm_data.core.config import BoundaryConditionConfig, FieldExpressionConfig
from plm_data.core.mesh import DomainGeometry
from plm_data.core.spatial_fields import (
    component_expressions,
    component_labels_for_dim,
    resolve_param_ref,
    scalar_expression_to_config,
)


def _as_3d_vector(value, gdim: int):
    """Embed a 2D tangential quantity in 3D for cross products."""
    if gdim == 2:
        return ufl.as_vector((value[0], value[1], 0))
    return value


def tangential_inner(u, v, n, gdim: int):
    """Return the tangential inner product on a boundary."""
    u_3d = _as_3d_vector(u, gdim)
    v_3d = _as_3d_vector(v, gdim)
    n_3d = _as_3d_vector(n, gdim)
    return ufl.inner(ufl.cross(u_3d, n_3d), ufl.cross(v_3d, n_3d))


def apply_maxwell_dirichlet_bcs(
    V: fem.FunctionSpace,
    domain_geom: DomainGeometry,
    bc_configs: dict[str, BoundaryConditionConfig],
    parameters: dict[str, float],
) -> list[fem.DirichletBC]:
    """Apply strong PEC-style Maxwell boundary conditions."""
    _validate_boundary_types(bc_configs, allowed={"dirichlet", "absorbing"})
    return apply_vector_dirichlet_bcs(V, domain_geom, bc_configs, parameters)


def build_absorbing_boundary_form(
    u,
    v,
    domain_geom: DomainGeometry,
    bc_configs: dict[str, BoundaryConditionConfig],
    coefficient,
    parameters: dict[str, float],
):
    """Build homogeneous absorbing boundary contributions."""
    _validate_boundary_types(bc_configs, allowed={"dirichlet", "absorbing"})

    gdim = domain_geom.mesh.geometry.dim
    n = ufl.FacetNormal(domain_geom.mesh)
    form = None
    for name, bc in bc_configs.items():
        if bc.type != "absorbing":
            continue
        _require_zero_absorbing_value(name, bc.value, gdim, parameters)
        term = coefficient * tangential_inner(u, v, n, gdim) * domain_geom.ds(
            domain_geom.boundary_names[name]
        )
        form = term if form is None else form + term
    return form


def _validate_boundary_types(
    bc_configs: dict[str, BoundaryConditionConfig],
    allowed: set[str],
) -> None:
    for name, bc in bc_configs.items():
        if bc.type not in allowed:
            allowed_str = ", ".join(sorted(allowed))
            raise ValueError(
                f"Maxwell boundary '{name}' uses unsupported type '{bc.type}'. "
                f"Allowed types: {allowed_str}."
            )


def _require_zero_absorbing_value(
    name: str,
    expr: FieldExpressionConfig,
    gdim: int,
    parameters: dict[str, float],
) -> None:
    """Ensure v1 absorbing boundaries are homogeneous."""
    components = component_expressions(expr, gdim)
    for label in component_labels_for_dim(gdim):
        component = scalar_expression_to_config(components[label])
        expr_type = component["type"]
        if expr_type in {"none", "zero"}:
            continue
        if expr_type == "constant":
            value = resolve_param_ref(component["params"]["value"], parameters)
            if value == 0.0:
                continue
        raise ValueError(
            f"Absorbing boundary '{name}' must use a zero value in v1. "
            f"Component '{label}' was configured with '{expr_type}'."
        )

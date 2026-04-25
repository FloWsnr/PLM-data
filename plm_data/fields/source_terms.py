"""Source term construction from field expression configs."""

import ufl
from dolfinx import mesh as dmesh

from plm_data.core.runtime_config import FieldExpressionConfig
from plm_data.fields.expressions import (
    component_expressions,
    component_labels_for_dim,
    resolve_param_ref,
    scalar_expression_to_config,
)
from plm_data.fields.ufl import build_ufl_field


def _is_trivially_zero_scalar_field(
    source_config: FieldExpressionConfig,
    parameters: dict[str, float],
) -> bool:
    """Return whether a scalar field config contributes identically zero."""
    if source_config.is_componentwise:
        raise ValueError("Expected a scalar field expression")
    if source_config.type in ("none", "zero", "custom"):
        return True
    if source_config.type == "constant":
        return resolve_param_ref(source_config.params["value"], parameters) == 0.0
    return False


def build_source_form(
    v: ufl.Argument,
    msh: dmesh.Mesh,
    source_config: FieldExpressionConfig,
    parameters: dict[str, float],
) -> ufl.Form | None:
    """Build a scalar source term contribution."""
    if source_config.is_componentwise:
        raise ValueError("build_source_form expects a scalar source config")

    if _is_trivially_zero_scalar_field(source_config, parameters):
        return None

    f = build_ufl_field(msh, scalar_expression_to_config(source_config), parameters)
    return ufl.inner(f, v) * ufl.dx


def build_vector_source_form(
    v: ufl.Argument,
    msh: dmesh.Mesh,
    source_config: FieldExpressionConfig,
    parameters: dict[str, float],
) -> ufl.Form | None:
    """Build a vector source term from a vector field config."""
    if not source_config.is_componentwise and source_config.type in (
        "none",
        "zero",
        "custom",
    ):
        return None

    gdim = msh.geometry.dim
    components = component_expressions(source_config, gdim)

    component_exprs = []
    has_nonzero = False
    for label in component_labels_for_dim(gdim):
        component_config = components[label]
        if _is_trivially_zero_scalar_field(component_config, parameters):
            component_exprs.append(ufl.as_ufl(0.0))
            continue

        component_exprs.append(
            build_ufl_field(
                msh,
                scalar_expression_to_config(component_config),
                parameters,
            )
        )
        has_nonzero = True

    if not has_nonzero:
        return None

    return ufl.inner(ufl.as_vector(component_exprs), v) * ufl.dx

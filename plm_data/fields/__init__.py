"""Shared field-expression runtime helpers."""

from plm_data.fields.expressions import (
    component_expressions,
    component_labels_for_dim,
    is_exact_zero_field_expression,
    normalize_field_config,
    resolve_param_ref,
    scalar_expression_to_config,
)
from plm_data.fields.interpolation import build_interpolator, build_vector_interpolator
from plm_data.fields.source_terms import build_source_form, build_vector_source_form
from plm_data.fields.ufl import build_ufl_field, build_vector_ufl_field

__all__ = [
    "build_interpolator",
    "build_source_form",
    "build_ufl_field",
    "build_vector_interpolator",
    "build_vector_source_form",
    "build_vector_ufl_field",
    "component_expressions",
    "component_labels_for_dim",
    "is_exact_zero_field_expression",
    "normalize_field_config",
    "resolve_param_ref",
    "scalar_expression_to_config",
]

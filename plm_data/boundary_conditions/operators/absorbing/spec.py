"""Absorbing boundary-operator spec."""

from plm_data.boundary_conditions.base import (
    BoundaryOperatorSpec,
    register_boundary_operator_spec,
)

OPERATOR_SPEC = register_boundary_operator_spec(
    BoundaryOperatorSpec(
        name="absorbing",
        value_shape="field",
        description="Absorbing Maxwell boundary condition.",
        allowed_field_shapes=("vector",),
    )
)

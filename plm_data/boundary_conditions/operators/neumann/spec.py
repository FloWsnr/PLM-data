"""Neumann boundary-operator spec."""

from plm_data.boundary_conditions.base import (
    BoundaryOperatorSpec,
    register_boundary_operator_spec,
)

OPERATOR_SPEC = register_boundary_operator_spec(
    BoundaryOperatorSpec(
        name="neumann",
        value_shape="field",
        description="Natural flux or traction boundary condition.",
    )
)

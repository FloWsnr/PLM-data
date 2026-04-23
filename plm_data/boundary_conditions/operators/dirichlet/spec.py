"""Dirichlet boundary-operator spec."""

from plm_data.boundary_conditions.base import (
    BoundaryOperatorSpec,
    register_boundary_operator_spec,
)

OPERATOR_SPEC = register_boundary_operator_spec(
    BoundaryOperatorSpec(
        name="dirichlet",
        value_shape="field",
        description="Strong value boundary condition.",
    )
)

"""Periodic boundary-operator spec."""

from plm_data.boundary_conditions.base import (
    BoundaryOperatorSpec,
    register_boundary_operator_spec,
)

OPERATOR_SPEC = register_boundary_operator_spec(
    BoundaryOperatorSpec(
        name="periodic",
        value_shape=None,
        requires_pair_with=True,
        description="Periodic side-pair constraint.",
    )
)

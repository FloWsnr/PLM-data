"""Simply-supported boundary-operator spec."""

from plm_data.boundary_conditions.base import (
    BoundaryOperatorSpec,
    register_boundary_operator_spec,
)

OPERATOR_SPEC = register_boundary_operator_spec(
    BoundaryOperatorSpec(
        name="simply_supported",
        value_shape=None,
        description="Homogeneous simply supported wall condition.",
        allowed_field_shapes=("scalar",),
    )
)

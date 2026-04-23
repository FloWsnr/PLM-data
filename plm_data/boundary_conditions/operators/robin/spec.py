"""Robin boundary-operator spec."""

from plm_data.boundary_conditions.base import (
    BoundaryOperatorParameterSpec,
    BoundaryOperatorSpec,
    register_boundary_operator_spec,
)

OPERATOR_SPEC = register_boundary_operator_spec(
    BoundaryOperatorSpec(
        name="robin",
        value_shape="field",
        description="Scalar Robin boundary condition.",
        allowed_field_shapes=("scalar",),
        parameter_specs={
            "alpha": BoundaryOperatorParameterSpec(
                name="alpha",
                kind="float",
                hard_min=0.0,
                sampling_min=0.0,
                sampling_max=10.0,
                description="Robin exchange coefficient multiplying the trial field.",
            ),
        },
    )
)

"""No-op initial-condition operator spec."""

from plm_data.initial_conditions.base import (
    InitialConditionOperatorSpec,
    register_initial_condition_operator_spec,
)

OPERATOR_SPEC = register_initial_condition_operator_spec(
    InitialConditionOperatorSpec(
        name="none",
        description="Leave the target field unset by the shared IC machinery.",
        parameters={},
        supports_vector_field_level=True,
    )
)

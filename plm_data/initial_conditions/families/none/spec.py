"""No-op initial-condition family spec."""

from plm_data.initial_conditions.base import (
    InitialConditionSpec,
    register_initial_condition_spec,
)

FAMILY_SPEC = register_initial_condition_spec(
    InitialConditionSpec(
        name="none",
        description="Leave the target field unset by the shared IC machinery.",
        parameters={},
        supports_vector_field_level=True,
    )
)

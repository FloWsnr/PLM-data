"""Custom initial-condition operator spec."""

from plm_data.initial_conditions.base import (
    InitialConditionOperatorSpec,
    register_initial_condition_operator_spec,
)

OPERATOR_SPEC = register_initial_condition_operator_spec(
    InitialConditionOperatorSpec(
        name="custom",
        description="Delegate initialization to PDE-specific code.",
        parameters={},
        supports_vector_field_level=True,
    )
)

"""Custom initial-condition family spec."""

from plm_data.initial_conditions.base import (
    InitialConditionSpec,
    register_initial_condition_spec,
)

FAMILY_SPEC = register_initial_condition_spec(
    InitialConditionSpec(
        name="custom",
        description="Delegate initialization to preset-specific code.",
        parameters={},
        supports_vector_field_level=True,
    )
)

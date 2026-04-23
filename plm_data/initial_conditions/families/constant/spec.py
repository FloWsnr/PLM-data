"""Constant initial-condition family spec."""

from plm_data.initial_conditions.base import (
    InitialConditionParameterSpec,
    InitialConditionSpec,
    register_initial_condition_spec,
)

FAMILY_SPEC = register_initial_condition_spec(
    InitialConditionSpec(
        name="constant",
        description="Initialize the field to one scalar value.",
        parameters={
            "value": InitialConditionParameterSpec(
                name="value",
                kind="float",
                description="Constant scalar value.",
            ),
        },
        common_scalar_family=True,
    )
)

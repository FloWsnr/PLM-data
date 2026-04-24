"""Constant initial-condition operator spec."""

from plm_data.initial_conditions.base import (
    InitialConditionOperatorParameterSpec,
    InitialConditionOperatorSpec,
    register_initial_condition_operator_spec,
)

OPERATOR_SPEC = register_initial_condition_operator_spec(
    InitialConditionOperatorSpec(
        name="constant",
        description="Initialize the field to one scalar value.",
        parameters={
            "value": InitialConditionOperatorParameterSpec(
                name="value",
                kind="float",
                description="Constant scalar value.",
            ),
        },
        common_scalar_operator=True,
    )
)

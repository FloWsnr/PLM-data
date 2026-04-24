"""Quadrants initial-condition operator spec."""

from plm_data.initial_conditions.base import (
    InitialConditionOperatorParameterSpec,
    InitialConditionOperatorSpec,
    register_initial_condition_operator_spec,
)

OPERATOR_SPEC = register_initial_condition_operator_spec(
    InitialConditionOperatorSpec(
        name="quadrants",
        description="Initialize piecewise-constant values on binary coordinate regions.",
        parameters={
            "split": InitialConditionOperatorParameterSpec(
                name="split",
                kind="coordinate_vector",
                description="Split coordinate for each active axis.",
            ),
            "region_values": InitialConditionOperatorParameterSpec(
                name="region_values",
                kind="region_value_map",
                description="Map from binary region key to scalar value.",
            ),
        },
        common_scalar_operator=True,
    )
)

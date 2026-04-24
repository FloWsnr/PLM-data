"""Step initial-condition operator spec."""

from plm_data.initial_conditions.base import (
    InitialConditionOperatorParameterSpec,
    InitialConditionOperatorSpec,
    register_initial_condition_operator_spec,
)

OPERATOR_SPEC = register_initial_condition_operator_spec(
    InitialConditionOperatorSpec(
        name="step",
        description="Initialize with two values separated by one axis-aligned split.",
        parameters={
            "value_left": InitialConditionOperatorParameterSpec(
                name="value_left",
                kind="float",
                description="Value below the split.",
            ),
            "value_right": InitialConditionOperatorParameterSpec(
                name="value_right",
                kind="float",
                description="Value above the split.",
            ),
            "x_split": InitialConditionOperatorParameterSpec(
                name="x_split",
                kind="float",
                description="Split location along the selected axis.",
            ),
            "axis": InitialConditionOperatorParameterSpec(
                name="axis",
                kind="int",
                hard_min=0,
                hard_max=2,
                description="Axis index used for the split.",
            ),
        },
        common_scalar_operator=True,
    )
)

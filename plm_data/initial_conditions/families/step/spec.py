"""Step initial-condition family spec."""

from plm_data.initial_conditions.base import (
    InitialConditionParameterSpec,
    InitialConditionSpec,
    register_initial_condition_spec,
)

FAMILY_SPEC = register_initial_condition_spec(
    InitialConditionSpec(
        name="step",
        description="Initialize with two values separated by one axis-aligned split.",
        parameters={
            "value_left": InitialConditionParameterSpec(
                name="value_left",
                kind="float",
                description="Value below the split.",
            ),
            "value_right": InitialConditionParameterSpec(
                name="value_right",
                kind="float",
                description="Value above the split.",
            ),
            "x_split": InitialConditionParameterSpec(
                name="x_split",
                kind="float",
                description="Split location along the selected axis.",
            ),
            "axis": InitialConditionParameterSpec(
                name="axis",
                kind="int",
                hard_min=0,
                hard_max=2,
                description="Axis index used for the split.",
            ),
        },
        common_scalar_family=True,
    )
)

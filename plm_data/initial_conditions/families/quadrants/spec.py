"""Quadrants initial-condition family spec."""

from plm_data.initial_conditions.base import (
    InitialConditionParameterSpec,
    InitialConditionSpec,
    register_initial_condition_spec,
)

FAMILY_SPEC = register_initial_condition_spec(
    InitialConditionSpec(
        name="quadrants",
        description="Initialize piecewise-constant values on binary coordinate regions.",
        parameters={
            "split": InitialConditionParameterSpec(
                name="split",
                kind="coordinate_vector",
                description="Split coordinate for each active axis.",
            ),
            "region_values": InitialConditionParameterSpec(
                name="region_values",
                kind="region_value_map",
                description="Map from binary region key to scalar value.",
            ),
        },
        common_scalar_family=True,
    )
)

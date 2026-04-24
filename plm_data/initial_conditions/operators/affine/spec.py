"""Affine initial-condition operator spec."""

from plm_data.initial_conditions.base import (
    InitialConditionOperatorParameterSpec,
    InitialConditionOperatorSpec,
    register_initial_condition_operator_spec,
)

OPERATOR_SPEC = register_initial_condition_operator_spec(
    InitialConditionOperatorSpec(
        name="affine",
        description="Initialize with a dimension-dependent affine scalar field.",
        parameters={
            "constant": InitialConditionOperatorParameterSpec(
                name="constant",
                kind="float",
                required=False,
                description="Constant offset.",
            ),
            "x": InitialConditionOperatorParameterSpec(
                name="x",
                kind="float",
                required=False,
                description="Coefficient for the x coordinate.",
            ),
            "y": InitialConditionOperatorParameterSpec(
                name="y",
                kind="float",
                required=False,
                description="Coefficient for the y coordinate when present.",
            ),
            "z": InitialConditionOperatorParameterSpec(
                name="z",
                kind="float",
                required=False,
                description="Coefficient for the z coordinate when present.",
            ),
        },
    )
)

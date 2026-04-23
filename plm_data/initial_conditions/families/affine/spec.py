"""Affine initial-condition family spec."""

from plm_data.initial_conditions.base import (
    InitialConditionParameterSpec,
    InitialConditionSpec,
    register_initial_condition_spec,
)

FAMILY_SPEC = register_initial_condition_spec(
    InitialConditionSpec(
        name="affine",
        description="Initialize with a dimension-dependent affine scalar field.",
        parameters={
            "constant": InitialConditionParameterSpec(
                name="constant",
                kind="float",
                required=False,
                description="Constant offset.",
            ),
            "x": InitialConditionParameterSpec(
                name="x",
                kind="float",
                required=False,
                description="Coefficient for the x coordinate.",
            ),
            "y": InitialConditionParameterSpec(
                name="y",
                kind="float",
                required=False,
                description="Coefficient for the y coordinate when present.",
            ),
            "z": InitialConditionParameterSpec(
                name="z",
                kind="float",
                required=False,
                description="Coefficient for the z coordinate when present.",
            ),
        },
    )
)

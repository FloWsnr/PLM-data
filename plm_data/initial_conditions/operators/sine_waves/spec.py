"""Sine-waves initial-condition operator spec."""

from plm_data.initial_conditions.base import (
    InitialConditionOperatorParameterSpec,
    InitialConditionOperatorSpec,
    register_initial_condition_operator_spec,
)

OPERATOR_SPEC = register_initial_condition_operator_spec(
    InitialConditionOperatorSpec(
        name="sine_waves",
        description="Initialize with one or more separable sine-wave modes.",
        parameters={
            "background": InitialConditionOperatorParameterSpec(
                name="background",
                kind="float",
                description="Background value.",
            ),
            "modes": InitialConditionOperatorParameterSpec(
                name="modes",
                kind="mode_list",
                description=(
                    "List of modes with amplitude, cycles, phase, and optional angle."
                ),
            ),
        },
        common_scalar_operator=True,
    )
)

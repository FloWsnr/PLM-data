"""Sine-waves initial-condition family spec."""

from plm_data.initial_conditions.base import (
    InitialConditionParameterSpec,
    InitialConditionSpec,
    register_initial_condition_spec,
)

FAMILY_SPEC = register_initial_condition_spec(
    InitialConditionSpec(
        name="sine_waves",
        description="Initialize with one or more separable sine-wave modes.",
        parameters={
            "background": InitialConditionParameterSpec(
                name="background",
                kind="float",
                description="Background value.",
            ),
            "modes": InitialConditionParameterSpec(
                name="modes",
                kind="mode_list",
                description=(
                    "List of modes with amplitude, cycles, phase, and optional angle."
                ),
            ),
        },
        common_scalar_family=True,
    )
)

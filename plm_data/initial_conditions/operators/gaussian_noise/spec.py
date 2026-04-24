"""Gaussian-noise initial-condition operator spec."""

from plm_data.initial_conditions.base import (
    InitialConditionOperatorParameterSpec,
    InitialConditionOperatorSpec,
    register_initial_condition_operator_spec,
)

OPERATOR_SPEC = register_initial_condition_operator_spec(
    InitialConditionOperatorSpec(
        name="gaussian_noise",
        description="Initialize with independent Gaussian noise at interpolation points.",
        parameters={
            "mean": InitialConditionOperatorParameterSpec(
                name="mean",
                kind="float",
                description="Noise mean.",
            ),
            "std": InitialConditionOperatorParameterSpec(
                name="std",
                kind="float",
                hard_min=0.0,
                description="Noise standard deviation.",
            ),
        },
        requires_seed=True,
        common_scalar_operator=True,
    )
)

"""Gaussian-noise initial-condition family spec."""

from plm_data.initial_conditions.base import (
    InitialConditionParameterSpec,
    InitialConditionSpec,
    register_initial_condition_spec,
)

FAMILY_SPEC = register_initial_condition_spec(
    InitialConditionSpec(
        name="gaussian_noise",
        description="Initialize with independent Gaussian noise at interpolation points.",
        parameters={
            "mean": InitialConditionParameterSpec(
                name="mean",
                kind="float",
                description="Noise mean.",
            ),
            "std": InitialConditionParameterSpec(
                name="std",
                kind="float",
                hard_min=0.0,
                description="Noise standard deviation.",
            ),
        },
        requires_seed=True,
        common_scalar_family=True,
    )
)

"""Gaussian-blobs initial-condition operator spec."""

from plm_data.initial_conditions.base import (
    InitialConditionOperatorParameterSpec,
    InitialConditionOperatorSpec,
    register_initial_condition_operator_spec,
)

OPERATOR_SPEC = register_initial_condition_operator_spec(
    InitialConditionOperatorSpec(
        name="gaussian_blobs",
        description="Initialize with one or more sampled anisotropic Gaussian blobs.",
        parameters={
            "background": InitialConditionOperatorParameterSpec(
                name="background",
                kind="float",
                description="Background value outside blobs.",
            ),
            "generators": InitialConditionOperatorParameterSpec(
                name="generators",
                kind="generator_list",
                description=(
                    "List of blob generators with count, amplitude, sigma, "
                    "center, and aspect_ratio."
                ),
            ),
        },
        requires_seed=True,
        common_scalar_operator=True,
    )
)

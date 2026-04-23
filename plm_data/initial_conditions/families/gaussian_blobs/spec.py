"""Gaussian-blobs initial-condition family spec."""

from plm_data.initial_conditions.base import (
    InitialConditionParameterSpec,
    InitialConditionSpec,
    register_initial_condition_spec,
)

FAMILY_SPEC = register_initial_condition_spec(
    InitialConditionSpec(
        name="gaussian_blobs",
        description="Initialize with one or more sampled anisotropic Gaussian blobs.",
        parameters={
            "background": InitialConditionParameterSpec(
                name="background",
                kind="float",
                description="Background value outside blobs.",
            ),
            "generators": InitialConditionParameterSpec(
                name="generators",
                kind="generator_list",
                description=(
                    "List of blob generators with count, amplitude, sigma, "
                    "center, and aspect_ratio."
                ),
            ),
        },
        requires_seed=True,
        common_scalar_family=True,
    )
)

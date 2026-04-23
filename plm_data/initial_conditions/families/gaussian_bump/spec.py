"""Gaussian-bump initial-condition family spec."""

from plm_data.initial_conditions.base import (
    InitialConditionParameterSpec,
    InitialConditionSpec,
    register_initial_condition_spec,
)

FAMILY_SPEC = register_initial_condition_spec(
    InitialConditionSpec(
        name="gaussian_bump",
        description="Initialize with one isotropic Gaussian bump.",
        parameters={
            "amplitude": InitialConditionParameterSpec(
                name="amplitude",
                kind="float",
                description="Peak amplitude.",
            ),
            "sigma": InitialConditionParameterSpec(
                name="sigma",
                kind="float",
                hard_min=0.0,
                description="Gaussian width.",
            ),
            "center": InitialConditionParameterSpec(
                name="center",
                kind="coordinate_vector",
                description="Bump center in domain coordinates.",
            ),
        },
        common_scalar_family=True,
    )
)

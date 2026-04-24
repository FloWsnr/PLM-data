"""Gaussian-bump initial-condition operator spec."""

from plm_data.initial_conditions.base import (
    InitialConditionOperatorParameterSpec,
    InitialConditionOperatorSpec,
    register_initial_condition_operator_spec,
)

OPERATOR_SPEC = register_initial_condition_operator_spec(
    InitialConditionOperatorSpec(
        name="gaussian_bump",
        description="Initialize with one isotropic Gaussian bump.",
        parameters={
            "amplitude": InitialConditionOperatorParameterSpec(
                name="amplitude",
                kind="float",
                description="Peak amplitude.",
            ),
            "sigma": InitialConditionOperatorParameterSpec(
                name="sigma",
                kind="float",
                hard_min=0.0,
                description="Gaussian width.",
            ),
            "center": InitialConditionOperatorParameterSpec(
                name="center",
                kind="coordinate_vector",
                description="Bump center in domain coordinates.",
            ),
        },
        common_scalar_operator=True,
    )
)

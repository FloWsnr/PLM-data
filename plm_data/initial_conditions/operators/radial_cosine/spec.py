"""Radial-cosine initial-condition operator spec."""

from plm_data.initial_conditions.base import (
    InitialConditionOperatorParameterSpec,
    InitialConditionOperatorSpec,
    register_initial_condition_operator_spec,
)

OPERATOR_SPEC = register_initial_condition_operator_spec(
    InitialConditionOperatorSpec(
        name="radial_cosine",
        description="Initialize with a cosine profile measured from a center.",
        parameters={
            "base": InitialConditionOperatorParameterSpec(
                name="base",
                kind="float",
                description="Background value.",
            ),
            "amplitude": InitialConditionOperatorParameterSpec(
                name="amplitude",
                kind="float",
                description="Cosine amplitude.",
            ),
            "frequency": InitialConditionOperatorParameterSpec(
                name="frequency",
                kind="float",
                description="Radial oscillation frequency.",
            ),
            "center": InitialConditionOperatorParameterSpec(
                name="center",
                kind="coordinate_vector",
                description="Profile center in domain coordinates.",
            ),
        },
        common_scalar_operator=True,
    )
)

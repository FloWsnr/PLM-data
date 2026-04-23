"""Radial-cosine initial-condition family spec."""

from plm_data.initial_conditions.base import (
    InitialConditionParameterSpec,
    InitialConditionSpec,
    register_initial_condition_spec,
)

FAMILY_SPEC = register_initial_condition_spec(
    InitialConditionSpec(
        name="radial_cosine",
        description="Initialize with a cosine profile measured from a center.",
        parameters={
            "base": InitialConditionParameterSpec(
                name="base",
                kind="float",
                description="Background value.",
            ),
            "amplitude": InitialConditionParameterSpec(
                name="amplitude",
                kind="float",
                description="Cosine amplitude.",
            ),
            "frequency": InitialConditionParameterSpec(
                name="frequency",
                kind="float",
                description="Radial oscillation frequency.",
            ),
            "center": InitialConditionParameterSpec(
                name="center",
                kind="coordinate_vector",
                description="Profile center in domain coordinates.",
            ),
        },
        common_scalar_family=True,
    )
)

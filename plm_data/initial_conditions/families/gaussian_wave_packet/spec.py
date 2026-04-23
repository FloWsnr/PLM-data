"""Gaussian-wave-packet initial-condition family spec."""

from plm_data.initial_conditions.base import (
    InitialConditionParameterSpec,
    InitialConditionSpec,
    register_initial_condition_spec,
)

FAMILY_SPEC = register_initial_condition_spec(
    InitialConditionSpec(
        name="gaussian_wave_packet",
        description="Initialize with a Gaussian envelope modulating a cosine carrier.",
        parameters={
            "amplitude": InitialConditionParameterSpec(
                name="amplitude",
                kind="float",
                description="Envelope amplitude.",
            ),
            "sigma": InitialConditionParameterSpec(
                name="sigma",
                kind="float",
                hard_min=0.0,
                description="Envelope width.",
            ),
            "center": InitialConditionParameterSpec(
                name="center",
                kind="coordinate_vector",
                description="Packet center in domain coordinates.",
            ),
            "wavevector": InitialConditionParameterSpec(
                name="wavevector",
                kind="float_vector",
                description="Carrier wavevector.",
            ),
            "phase": InitialConditionParameterSpec(
                name="phase",
                kind="float",
                description="Carrier phase.",
            ),
        },
        common_scalar_family=True,
    )
)

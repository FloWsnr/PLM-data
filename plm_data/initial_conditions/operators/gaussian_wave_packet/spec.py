"""Gaussian-wave-packet initial-condition operator spec."""

from plm_data.initial_conditions.base import (
    InitialConditionOperatorParameterSpec,
    InitialConditionOperatorSpec,
    register_initial_condition_operator_spec,
)

OPERATOR_SPEC = register_initial_condition_operator_spec(
    InitialConditionOperatorSpec(
        name="gaussian_wave_packet",
        description="Initialize with a Gaussian envelope modulating a cosine carrier.",
        parameters={
            "amplitude": InitialConditionOperatorParameterSpec(
                name="amplitude",
                kind="float",
                description="Envelope amplitude.",
            ),
            "sigma": InitialConditionOperatorParameterSpec(
                name="sigma",
                kind="float",
                hard_min=0.0,
                description="Envelope width.",
            ),
            "center": InitialConditionOperatorParameterSpec(
                name="center",
                kind="coordinate_vector",
                description="Packet center in domain coordinates.",
            ),
            "wavevector": InitialConditionOperatorParameterSpec(
                name="wavevector",
                kind="float_vector",
                description="Carrier wavevector.",
            ),
            "phase": InitialConditionOperatorParameterSpec(
                name="phase",
                kind="float",
                description="Carrier phase.",
            ),
        },
        common_scalar_operator=True,
    )
)

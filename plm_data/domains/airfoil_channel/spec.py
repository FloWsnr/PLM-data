"""Sampling and validation metadata for the airfoil-channel domain."""

from plm_data.domains.base import (
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)
from plm_data.domains.validators import validate_airfoil_channel_params


DOMAIN_SPEC = register_domain_spec(
    DomainSpec(
        name="airfoil_channel",
        dimension=2,
        description="Rectangular channel with a symmetric NACA-style airfoil cutout.",
        parameters={
            "length": DomainParameterSpec(
                name="length",
                kind="float",
                hard_min=0.0,
                sampling_min=2.0,
                sampling_max=5.0,
                description="Channel length.",
            ),
            "height": DomainParameterSpec(
                name="height",
                kind="float",
                hard_min=0.0,
                sampling_min=0.8,
                sampling_max=2.0,
                description="Channel height.",
            ),
            "airfoil_center": DomainParameterSpec(
                name="airfoil_center",
                kind="float_vector",
                length=2,
                description="Airfoil center.",
            ),
            "chord_length": DomainParameterSpec(
                name="chord_length",
                kind="float",
                hard_min=0.0,
                sampling_min=0.3,
                sampling_max=1.2,
                description="Airfoil chord length.",
            ),
            "thickness_ratio": DomainParameterSpec(
                name="thickness_ratio",
                kind="float",
                hard_min=0.0,
                hard_max=0.25,
                sampling_min=0.08,
                sampling_max=0.18,
                description="Airfoil thickness-to-chord ratio.",
            ),
            "attack_angle_degrees": DomainParameterSpec(
                name="attack_angle_degrees",
                kind="float",
                hard_min=-35.0,
                hard_max=35.0,
                sampling_min=-15.0,
                sampling_max=15.0,
                description="Airfoil angle of attack.",
            ),
            "mesh_size": DomainParameterSpec(
                name="mesh_size",
                kind="float",
                hard_min=0.0,
                sampling_min=0.03,
                sampling_max=0.12,
                description="Target Gmsh mesh size.",
            ),
        },
        boundary_names=("inlet", "outlet", "walls", "airfoil"),
        boundary_roles={
            "all": ("inlet", "outlet", "walls", "airfoil"),
            "inlet": ("inlet",),
            "outlet": ("outlet",),
            "walls": ("walls",),
            "airfoil": ("airfoil",),
            "solid": ("walls", "airfoil"),
            "open": ("inlet", "outlet"),
        },
        coordinate_regions=(
            "interior",
            "upstream",
            "downstream",
            "wake",
            "near_airfoil",
        ),
        validate_params=validate_airfoil_channel_params,
    )
)

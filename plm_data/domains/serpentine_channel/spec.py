"""Sampling and validation metadata for the serpentine-channel domain."""

from plm_data.domains.base import (
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)
from plm_data.domains.validators import validate_serpentine_channel_params


DOMAIN_SPEC = register_domain_spec(
    DomainSpec(
        name="serpentine_channel",
        dimension=2,
        description="Rounded orthogonal channel with alternating lane bends.",
        parameters={
            "channel_length": DomainParameterSpec(
                name="channel_length",
                kind="float",
                hard_min=0.0,
                sampling_min=0.8,
                sampling_max=2.5,
                description="Length of each straight lane.",
            ),
            "lane_spacing": DomainParameterSpec(
                name="lane_spacing",
                kind="float",
                hard_min=0.0,
                sampling_min=0.25,
                sampling_max=0.9,
                description="Centerline spacing between neighboring lanes.",
            ),
            "n_bends": DomainParameterSpec(
                name="n_bends",
                kind="int",
                hard_min=2,
                sampling_min=2,
                sampling_max=7,
                description="Number of lane reversals.",
            ),
            "channel_width": DomainParameterSpec(
                name="channel_width",
                kind="float",
                hard_min=0.0,
                sampling_min=0.12,
                sampling_max=0.35,
                description="Uniform channel width.",
            ),
            "mesh_size": DomainParameterSpec(
                name="mesh_size",
                kind="float",
                hard_min=0.0,
                sampling_min=0.03,
                sampling_max=0.14,
                description="Target Gmsh mesh size.",
            ),
        },
        boundary_names=("inlet", "outlet", "walls"),
        boundary_roles={
            "all": ("inlet", "outlet", "walls"),
            "inlet": ("inlet",),
            "outlet": ("outlet",),
            "walls": ("walls",),
            "solid": ("walls",),
            "open": ("inlet", "outlet"),
        },
        coordinate_regions=("interior", "inlet_lane", "outlet_lane", "bend"),
        validate_params=validate_serpentine_channel_params,
    )
)

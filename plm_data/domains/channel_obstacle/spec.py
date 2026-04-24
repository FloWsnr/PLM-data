"""Sampling and validation metadata for the channel-obstacle domain."""

from plm_data.domains.base import (
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)
from plm_data.domains.validators import validate_channel_obstacle_params


DOMAIN_SPEC = register_domain_spec(
    DomainSpec(
        name="channel_obstacle",
        dimension=2,
        description="Rectangular channel with a circular interior obstacle.",
        parameters={
            "length": DomainParameterSpec(
                name="length",
                kind="float",
                hard_min=0.0,
                sampling_min=1.5,
                sampling_max=4.0,
                description="Channel length.",
            ),
            "height": DomainParameterSpec(
                name="height",
                kind="float",
                hard_min=0.0,
                sampling_min=0.7,
                sampling_max=1.8,
                description="Channel height.",
            ),
            "obstacle_center": DomainParameterSpec(
                name="obstacle_center",
                kind="float_vector",
                length=2,
                description="Circular obstacle center.",
            ),
            "obstacle_radius": DomainParameterSpec(
                name="obstacle_radius",
                kind="float",
                hard_min=0.0,
                sampling_min=0.06,
                sampling_max=0.25,
                description="Circular obstacle radius.",
            ),
            "mesh_size": DomainParameterSpec(
                name="mesh_size",
                kind="float",
                hard_min=0.0,
                sampling_min=0.03,
                sampling_max=0.16,
                description="Target Gmsh mesh size.",
            ),
        },
        boundary_names=("inlet", "outlet", "walls", "obstacle"),
        boundary_roles={
            "all": ("inlet", "outlet", "walls", "obstacle"),
            "inlet": ("inlet",),
            "outlet": ("outlet",),
            "walls": ("walls",),
            "obstacle": ("obstacle",),
            "solid": ("walls", "obstacle"),
            "open": ("inlet", "outlet"),
        },
        coordinate_regions=(
            "interior",
            "upstream",
            "downstream",
            "wake",
            "near_obstacle",
        ),
        validate_params=validate_channel_obstacle_params,
    )
)

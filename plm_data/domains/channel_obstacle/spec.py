"""Sampling and validation metadata for the channel-obstacle domain."""

from plm_data.domains.base import (
    COMMON_BOUNDARY_FAMILIES,
    COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)


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
        allowed_boundary_families=COMMON_BOUNDARY_FAMILIES
        + ("open_channel", "no_slip_obstacle", "inlet_outlet_drive"),
        allowed_initial_condition_families=COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
        coordinate_regions=(
            "interior",
            "upstream",
            "downstream",
            "wake",
            "near_obstacle",
        ),
    )
)

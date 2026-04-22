"""Sampling and validation metadata for the venturi-channel domain."""

from plm_data.domains.base import (
    COMMON_BOUNDARY_FAMILIES,
    COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)


DOMAIN_SPEC = register_domain_spec(
    DomainSpec(
        name="venturi_channel",
        dimension=2,
        description="Rectangular channel with a smooth constricted throat.",
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
                description="Full channel height.",
            ),
            "throat_height": DomainParameterSpec(
                name="throat_height",
                kind="float",
                hard_min=0.0,
                sampling_min=0.25,
                sampling_max=1.0,
                description="Minimum height at the throat.",
            ),
            "constriction_center_x": DomainParameterSpec(
                name="constriction_center_x",
                kind="float",
                hard_min=0.0,
                description="x-coordinate of the throat center.",
            ),
            "constriction_radius": DomainParameterSpec(
                name="constriction_radius",
                kind="float",
                hard_min=0.0,
                sampling_min=0.25,
                sampling_max=1.2,
                description="Radius of the circular wall cuts.",
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
        boundary_names=("inlet", "outlet", "walls"),
        boundary_roles={
            "all": ("inlet", "outlet", "walls"),
            "inlet": ("inlet",),
            "outlet": ("outlet",),
            "walls": ("walls",),
            "solid": ("walls",),
            "open": ("inlet", "outlet"),
        },
        allowed_boundary_families=COMMON_BOUNDARY_FAMILIES
        + ("open_channel", "inlet_outlet_drive"),
        allowed_initial_condition_families=COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
        coordinate_regions=("interior", "upstream", "downstream", "throat"),
    )
)

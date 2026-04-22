"""Sampling and validation metadata for the side-cavity channel domain."""

from plm_data.domains.base import (
    COMMON_BOUNDARY_FAMILIES,
    COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)


DOMAIN_SPEC = register_domain_spec(
    DomainSpec(
        name="side_cavity_channel",
        dimension=2,
        description="Rectangular channel with one rectangular side cavity.",
        parameters={
            "length": DomainParameterSpec(
                name="length",
                kind="float",
                hard_min=0.0,
                sampling_min=1.5,
                sampling_max=4.0,
                description="Main channel length.",
            ),
            "height": DomainParameterSpec(
                name="height",
                kind="float",
                hard_min=0.0,
                sampling_min=0.6,
                sampling_max=1.5,
                description="Main channel height.",
            ),
            "cavity_width": DomainParameterSpec(
                name="cavity_width",
                kind="float",
                hard_min=0.0,
                sampling_min=0.25,
                sampling_max=1.0,
                description="Width of the side cavity opening.",
            ),
            "cavity_depth": DomainParameterSpec(
                name="cavity_depth",
                kind="float",
                hard_min=0.0,
                sampling_min=0.15,
                sampling_max=0.8,
                description="Depth of the side cavity.",
            ),
            "cavity_center_x": DomainParameterSpec(
                name="cavity_center_x",
                kind="float",
                hard_min=0.0,
                description="x-coordinate of the cavity center.",
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
        allowed_boundary_families=COMMON_BOUNDARY_FAMILIES
        + ("open_channel", "cavity_drive", "inlet_outlet_drive"),
        allowed_initial_condition_families=COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
        coordinate_regions=("interior", "main_channel", "cavity", "cavity_mouth"),
    )
)

"""Sampling and validation metadata for the Y-bifurcation domain."""

from plm_data.domains.base import (
    COMMON_BOUNDARY_FAMILIES,
    COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)


DOMAIN_SPEC = register_domain_spec(
    DomainSpec(
        name="y_bifurcation",
        dimension=2,
        description="Y-shaped channel with one inlet and two outlet branches.",
        parameters={
            "inlet_length": DomainParameterSpec(
                name="inlet_length",
                kind="float",
                hard_min=0.0,
                sampling_min=0.6,
                sampling_max=2.0,
                description="Length of the incoming trunk.",
            ),
            "branch_length": DomainParameterSpec(
                name="branch_length",
                kind="float",
                hard_min=0.0,
                sampling_min=0.6,
                sampling_max=2.0,
                description="Length of each outgoing branch.",
            ),
            "branch_angle_degrees": DomainParameterSpec(
                name="branch_angle_degrees",
                kind="float",
                hard_min=0.0,
                hard_max=85.0,
                sampling_min=25.0,
                sampling_max=60.0,
                description="Branch angle away from the trunk centerline.",
            ),
            "channel_width": DomainParameterSpec(
                name="channel_width",
                kind="float",
                hard_min=0.0,
                sampling_min=0.12,
                sampling_max=0.45,
                description="Uniform channel width.",
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
        boundary_names=("inlet", "outlet_upper", "outlet_lower", "walls"),
        boundary_roles={
            "all": ("inlet", "outlet_upper", "outlet_lower", "walls"),
            "inlet": ("inlet",),
            "outlets": ("outlet_upper", "outlet_lower"),
            "outlet_upper": ("outlet_upper",),
            "outlet_lower": ("outlet_lower",),
            "walls": ("walls",),
            "solid": ("walls",),
            "open": ("inlet", "outlet_upper", "outlet_lower"),
        },
        allowed_boundary_families=COMMON_BOUNDARY_FAMILIES
        + ("open_channel", "branch_drive", "inlet_outlet_drive"),
        allowed_initial_condition_families=COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
        coordinate_regions=(
            "interior",
            "trunk",
            "upper_branch",
            "lower_branch",
            "junction",
        ),
    )
)

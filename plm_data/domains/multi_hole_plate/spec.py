"""Sampling and validation metadata for the multi-hole plate domain."""

from plm_data.domains.base import (
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)
from plm_data.domains.validators import validate_multi_hole_plate_params


DOMAIN_SPEC = register_domain_spec(
    DomainSpec(
        name="multi_hole_plate",
        dimension=2,
        description=(
            "Rectangular plate with one or more circular interior holes. Hole "
            "boundaries may be grouped as 'holes' or named individually."
        ),
        parameters={
            "width": DomainParameterSpec(
                name="width",
                kind="float",
                hard_min=0.0,
                sampling_min=1.0,
                sampling_max=3.0,
                description="Plate width.",
            ),
            "height": DomainParameterSpec(
                name="height",
                kind="float",
                hard_min=0.0,
                sampling_min=0.7,
                sampling_max=2.0,
                description="Plate height.",
            ),
            "holes": DomainParameterSpec(
                name="holes",
                kind="hole_list",
                description="List of circular hole center/radius definitions.",
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
        boundary_names=("outer", "holes"),
        dynamic_boundary_patterns=("hole_*",),
        boundary_roles={
            "all": ("outer", "holes"),
            "outer": ("outer",),
            "holes": ("holes",),
            "solid": ("outer", "holes"),
        },
        coordinate_regions=("interior", "near_holes", "between_holes", "near_outer"),
        validate_params=validate_multi_hole_plate_params,
    )
)

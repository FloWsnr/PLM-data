"""Sampling and validation metadata for the L-shaped domain."""

from plm_data.domains.base import (
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)
from plm_data.domains.validators import validate_l_shape_params


DOMAIN_SPEC = register_domain_spec(
    DomainSpec(
        name="l_shape",
        dimension=2,
        description="Planar L-shape with outer and reentrant notch boundaries.",
        parameters={
            "outer_width": DomainParameterSpec(
                name="outer_width",
                kind="float",
                hard_min=0.0,
                sampling_min=0.8,
                sampling_max=2.0,
                description="Width of the bounding rectangle.",
            ),
            "outer_height": DomainParameterSpec(
                name="outer_height",
                kind="float",
                hard_min=0.0,
                sampling_min=0.8,
                sampling_max=2.0,
                description="Height of the bounding rectangle.",
            ),
            "cutout_width": DomainParameterSpec(
                name="cutout_width",
                kind="float",
                hard_min=0.0,
                sampling_min=0.2,
                sampling_max=0.9,
                description="Width of the removed top-right rectangle.",
            ),
            "cutout_height": DomainParameterSpec(
                name="cutout_height",
                kind="float",
                hard_min=0.0,
                sampling_min=0.2,
                sampling_max=0.9,
                description="Height of the removed top-right rectangle.",
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
        boundary_names=("outer", "notch"),
        boundary_roles={
            "all": ("outer", "notch"),
            "outer": ("outer",),
            "notch": ("notch",),
            "walls": ("outer", "notch"),
        },
        coordinate_regions=("interior", "lower_arm", "left_arm", "near_notch"),
        validate_params=validate_l_shape_params,
    )
)

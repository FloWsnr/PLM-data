"""Sampling and validation metadata for the dumbbell domain."""

from plm_data.domains.base import (
    COMMON_BOUNDARY_FAMILIES,
    COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)


DOMAIN_SPEC = register_domain_spec(
    DomainSpec(
        name="dumbbell",
        dimension=2,
        description="Two circular lobes joined by a rectangular neck.",
        parameters={
            "left_center": DomainParameterSpec(
                name="left_center",
                kind="float_vector",
                length=2,
                description="Center of the left lobe.",
            ),
            "right_center": DomainParameterSpec(
                name="right_center",
                kind="float_vector",
                length=2,
                description="Center of the right lobe.",
            ),
            "lobe_radius": DomainParameterSpec(
                name="lobe_radius",
                kind="float",
                hard_min=0.0,
                sampling_min=0.18,
                sampling_max=0.55,
                description="Radius of each lobe.",
            ),
            "neck_width": DomainParameterSpec(
                name="neck_width",
                kind="float",
                hard_min=0.0,
                sampling_min=0.06,
                sampling_max=0.35,
                description="Width of the bridge between lobes.",
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
        boundary_names=("outer",),
        boundary_roles={
            "all": ("outer",),
            "outer": ("outer",),
            "walls": ("outer",),
        },
        allowed_boundary_families=COMMON_BOUNDARY_FAMILIES,
        allowed_initial_condition_families=COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
        coordinate_regions=("interior", "left_lobe", "right_lobe", "neck"),
    )
)

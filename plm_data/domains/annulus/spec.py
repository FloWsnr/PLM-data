"""Sampling and validation metadata for the annulus domain."""

from plm_data.domains.base import (
    COMMON_BOUNDARY_FAMILIES,
    COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)


DOMAIN_SPEC = register_domain_spec(
    DomainSpec(
        name="annulus",
        dimension=2,
        description="Planar annulus with inner and outer circular boundaries.",
        parameters={
            "center": DomainParameterSpec(
                name="center",
                kind="float_vector",
                length=2,
                sampling_min=-1.0,
                sampling_max=1.0,
                description="Shared center of the inner and outer circles.",
            ),
            "inner_radius": DomainParameterSpec(
                name="inner_radius",
                kind="float",
                hard_min=0.0,
                sampling_min=0.15,
                sampling_max=0.75,
                description="Inner radius.",
            ),
            "outer_radius": DomainParameterSpec(
                name="outer_radius",
                kind="float",
                hard_min=0.0,
                sampling_min=0.6,
                sampling_max=1.6,
                description="Outer radius.",
            ),
            "mesh_size": DomainParameterSpec(
                name="mesh_size",
                kind="float",
                hard_min=0.0,
                sampling_min=0.03,
                sampling_max=0.18,
                description="Target Gmsh mesh size.",
            ),
        },
        boundary_names=("inner", "outer"),
        boundary_roles={
            "all": ("inner", "outer"),
            "inner": ("inner",),
            "outer": ("outer",),
            "walls": ("inner", "outer"),
        },
        allowed_boundary_families=COMMON_BOUNDARY_FAMILIES + ("inner_outer_drive",),
        allowed_initial_condition_families=COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
        coordinate_regions=("interior", "near_inner", "near_outer", "middle_ring"),
    )
)

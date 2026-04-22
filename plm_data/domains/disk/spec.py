"""Sampling and validation metadata for the disk domain."""

from plm_data.domains.base import (
    COMMON_BOUNDARY_FAMILIES,
    COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)


DOMAIN_SPEC = register_domain_spec(
    DomainSpec(
        name="disk",
        dimension=2,
        description="Planar disk with one outer boundary.",
        parameters={
            "center": DomainParameterSpec(
                name="center",
                kind="float_vector",
                length=2,
                sampling_min=-1.0,
                sampling_max=1.0,
                description="Disk center.",
            ),
            "radius": DomainParameterSpec(
                name="radius",
                kind="float",
                hard_min=0.0,
                sampling_min=0.35,
                sampling_max=1.5,
                description="Disk radius.",
            ),
            "mesh_size": DomainParameterSpec(
                name="mesh_size",
                kind="float",
                hard_min=0.0,
                sampling_min=0.03,
                sampling_max=0.2,
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
        coordinate_regions=("interior", "center", "near_outer", "radial_band"),
    )
)

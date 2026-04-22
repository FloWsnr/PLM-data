"""Sampling and validation metadata for the box domain."""

from plm_data.domains.base import (
    COMMON_BOUNDARY_FAMILIES,
    COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)


DOMAIN_SPEC = register_domain_spec(
    DomainSpec(
        name="box",
        dimension=3,
        description="Axis-aligned 3D box with six named side boundaries.",
        parameters={
            "size": DomainParameterSpec(
                name="size",
                kind="float_vector",
                length=3,
                hard_min=0.0,
                sampling_min=0.5,
                sampling_max=3.0,
                description="Domain side lengths [Lx, Ly, Lz].",
            ),
            "mesh_resolution": DomainParameterSpec(
                name="mesh_resolution",
                kind="int_vector",
                length=3,
                hard_min=1,
                sampling_min=12,
                sampling_max=96,
                description="Number of cells along each axis.",
            ),
        },
        boundary_names=("x-", "x+", "y-", "y+", "z-", "z+"),
        boundary_roles={
            "all": ("x-", "x+", "y-", "y+", "z-", "z+"),
            "x_min": ("x-",),
            "x_max": ("x+",),
            "y_min": ("y-",),
            "y_max": ("y+",),
            "z_min": ("z-",),
            "z_max": ("z+",),
            "x_pair": ("x-", "x+"),
            "y_pair": ("y-", "y+"),
            "z_pair": ("z-", "z+"),
            "walls": ("x-", "x+", "y-", "y+", "z-", "z+"),
        },
        periodic_pairs=(("x-", "x+"), ("y-", "y+"), ("z-", "z+")),
        allowed_boundary_families=COMMON_BOUNDARY_FAMILIES
        + ("periodic_axis", "full_periodic", "lid_driven_cavity"),
        allowed_initial_condition_families=COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
        coordinate_regions=("interior", "center", "lower_half", "upper_half"),
    )
)

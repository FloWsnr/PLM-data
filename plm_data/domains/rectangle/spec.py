"""Sampling and validation metadata for the rectangle domain."""

from plm_data.domains.base import (
    COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)


DOMAIN_SPEC = register_domain_spec(
    DomainSpec(
        name="rectangle",
        dimension=2,
        description="Axis-aligned 2D rectangle with four named side boundaries.",
        parameters={
            "size": DomainParameterSpec(
                name="size",
                kind="float_vector",
                length=2,
                hard_min=0.0,
                sampling_min=0.5,
                sampling_max=4.0,
                description="Domain side lengths [Lx, Ly].",
            ),
            "mesh_resolution": DomainParameterSpec(
                name="mesh_resolution",
                kind="int_vector",
                length=2,
                hard_min=1,
                sampling_min=32,
                sampling_max=256,
                description="Number of cells along each axis.",
            ),
        },
        boundary_names=("x-", "x+", "y-", "y+"),
        boundary_roles={
            "all": ("x-", "x+", "y-", "y+"),
            "x_min": ("x-",),
            "x_max": ("x+",),
            "y_min": ("y-",),
            "y_max": ("y+",),
            "x_pair": ("x-", "x+"),
            "y_pair": ("y-", "y+"),
            "walls": ("x-", "x+", "y-", "y+"),
        },
        periodic_pairs=(("x-", "x+"), ("y-", "y+")),
        allowed_boundary_families=(
            "all_dirichlet",
            "all_neumann",
            "all_robin",
            "periodic_axis",
            "full_periodic",
            "lid_driven_cavity",
        ),
        allowed_initial_condition_families=COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
        coordinate_regions=(
            "interior",
            "center",
            "left_half",
            "right_half",
            "lower_half",
            "upper_half",
        ),
    )
)

"""Sampling and validation metadata for the interval domain."""

from plm_data.domains.base import (
    COMMON_BOUNDARY_FAMILIES,
    COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)


DOMAIN_SPEC = register_domain_spec(
    DomainSpec(
        name="interval",
        dimension=1,
        description="One-dimensional interval with two endpoint boundaries.",
        parameters={
            "size": DomainParameterSpec(
                name="size",
                kind="float",
                hard_min=0.0,
                sampling_min=0.5,
                sampling_max=4.0,
                description="Interval length.",
            ),
            "mesh_resolution": DomainParameterSpec(
                name="mesh_resolution",
                kind="int",
                hard_min=1,
                sampling_min=64,
                sampling_max=512,
                description="Number of interval cells.",
            ),
        },
        boundary_names=("x-", "x+"),
        boundary_roles={
            "all": ("x-", "x+"),
            "x_min": ("x-",),
            "x_max": ("x+",),
            "x_pair": ("x-", "x+"),
        },
        periodic_pairs=(("x-", "x+"),),
        allowed_boundary_families=COMMON_BOUNDARY_FAMILIES
        + ("periodic_axis", "full_periodic"),
        allowed_initial_condition_families=COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
        coordinate_regions=("interior", "left_half", "right_half", "center"),
    )
)

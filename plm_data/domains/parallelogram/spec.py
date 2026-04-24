"""Sampling and validation metadata for the parallelogram domain."""

from plm_data.domains.base import (
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)
from plm_data.domains.validators import validate_parallelogram_params


DOMAIN_SPEC = register_domain_spec(
    DomainSpec(
        name="parallelogram",
        dimension=2,
        description="Affine image of the unit square with paired side boundaries.",
        parameters={
            "origin": DomainParameterSpec(
                name="origin",
                kind="float_vector",
                length=2,
                sampling_min=-1.0,
                sampling_max=1.0,
                description="Lower-left origin of the affine cell.",
            ),
            "axis_x": DomainParameterSpec(
                name="axis_x",
                kind="float_vector",
                length=2,
                description="First affine basis vector.",
            ),
            "axis_y": DomainParameterSpec(
                name="axis_y",
                kind="float_vector",
                length=2,
                description="Second affine basis vector.",
            ),
            "mesh_resolution": DomainParameterSpec(
                name="mesh_resolution",
                kind="int_vector",
                length=2,
                hard_min=1,
                sampling_min=32,
                sampling_max=256,
                description="Number of cells along each reference axis.",
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
        coordinate_regions=("interior", "center", "left_half", "right_half"),
        validate_params=validate_parallelogram_params,
    )
)

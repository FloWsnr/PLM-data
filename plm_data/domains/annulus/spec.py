"""Sampling and validation metadata for the annulus domain."""

import math

from plm_data.domains.base import (
    CoordinateSample,
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)
from plm_data.domains.validators import validate_annulus_params


def _uniform(context, name: str, minimum: float, maximum: float) -> float:
    value = float(context.child(name).rng.uniform(minimum, maximum))
    context.values[name] = value
    return value


def _annular_sample(
    context,
    domain,
    region: str,
    band_min: float,
    band_max: float,
) -> CoordinateSample:
    center = domain.params["center"]
    inner_radius = float(domain.params["inner_radius"])
    outer_radius = float(domain.params["outer_radius"])
    width = outer_radius - inner_radius
    angle = _uniform(context, f"domain.region.{region}.angle", 0.0, 2.0 * math.pi)
    radial = _uniform(
        context,
        f"domain.region.{region}.radial",
        inner_radius + band_min * width,
        inner_radius + band_max * width,
    )
    return CoordinateSample(
        point=[
            float(center[0]) + radial * math.cos(angle),
            float(center[1]) + radial * math.sin(angle),
        ],
        scale=width,
    )


def _sample_interior(context, domain) -> CoordinateSample:
    return _annular_sample(context, domain, "interior", 0.25, 0.75)


def _sample_near_inner(context, domain) -> CoordinateSample:
    return _annular_sample(context, domain, "near_inner", 0.05, 0.25)


def _sample_near_outer(context, domain) -> CoordinateSample:
    return _annular_sample(context, domain, "near_outer", 0.75, 0.95)


def _sample_middle_ring(context, domain) -> CoordinateSample:
    return _annular_sample(context, domain, "middle_ring", 0.4, 0.6)


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
        supported_boundary_scenarios=("scalar_all_neumann",),
        supported_initial_condition_scenarios=("heat_gaussian_bump",),
        coordinate_regions=("interior", "near_inner", "near_outer", "middle_ring"),
        validate_params=validate_annulus_params,
        coordinate_region_samplers={
            "interior": _sample_interior,
            "near_inner": _sample_near_inner,
            "near_outer": _sample_near_outer,
            "middle_ring": _sample_middle_ring,
        },
    )
)

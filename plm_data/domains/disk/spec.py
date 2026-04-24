"""Sampling and validation metadata for the disk domain."""

import math

from plm_data.domains.base import (
    CoordinateSample,
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)
from plm_data.domains.validators import validate_disk_params


def _uniform(context, name: str, minimum: float, maximum: float) -> float:
    value = float(context.child(name).rng.uniform(minimum, maximum))
    context.values[name] = value
    return value


def _radial_sample(
    context, domain, region: str, r_min: float, r_max: float
) -> CoordinateSample:
    center = domain.params["center"]
    radius = float(domain.params["radius"])
    angle = _uniform(context, f"domain.region.{region}.angle", 0.0, 2.0 * math.pi)
    radial = (
        _uniform(
            context,
            f"domain.region.{region}.radial_fraction",
            r_min,
            r_max,
        )
        * radius
    )
    return CoordinateSample(
        point=[
            float(center[0]) + radial * math.cos(angle),
            float(center[1]) + radial * math.sin(angle),
        ],
        scale=radius,
    )


def _sample_interior(context, domain) -> CoordinateSample:
    return _radial_sample(context, domain, "interior", 0.0, 0.55)


def _sample_center(context, domain) -> CoordinateSample:
    return _radial_sample(context, domain, "center", 0.0, 0.25)


def _sample_near_outer(context, domain) -> CoordinateSample:
    return _radial_sample(context, domain, "near_outer", 0.65, 0.9)


def _sample_radial_band(context, domain) -> CoordinateSample:
    return _radial_sample(context, domain, "radial_band", 0.35, 0.75)


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
        supported_boundary_scenarios=("scalar_all_neumann",),
        supported_initial_condition_scenarios=("heat_gaussian_bump",),
        coordinate_regions=("interior", "center", "near_outer", "radial_band"),
        validate_params=validate_disk_params,
        coordinate_region_samplers={
            "interior": _sample_interior,
            "center": _sample_center,
            "near_outer": _sample_near_outer,
            "radial_band": _sample_radial_band,
        },
    )
)

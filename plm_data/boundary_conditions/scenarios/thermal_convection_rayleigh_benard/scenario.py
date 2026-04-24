"""Rayleigh-Benard thermal-convection boundary scenario."""

from typing import TYPE_CHECKING

from plm_data.boundary_conditions.scenarios.base import (
    BoundaryScenario,
    BoundaryScenarioSpec,
    register_boundary_scenario,
)
from plm_data.core.runtime_config import (
    BoundaryConditionConfig,
    BoundaryFieldConfig,
    DomainConfig,
    FieldExpressionConfig,
)
from plm_data.domains.registry import get_domain_spec

if TYPE_CHECKING:
    from plm_data.sampling.context import SamplingContext


def _constant(value: float) -> FieldExpressionConfig:
    return FieldExpressionConfig(type="constant", params={"value": value})


def _vector_zero() -> FieldExpressionConfig:
    return FieldExpressionConfig(type="zero")


def _build(
    context: "SamplingContext",
    domain: DomainConfig,
) -> dict[str, BoundaryFieldConfig]:
    if context.pde_name != "thermal_convection":
        raise ValueError(
            "thermal_convection_rayleigh_benard supports only thermal_convection. "
            f"Got {context.pde_name!r}."
        )
    domain_spec = get_domain_spec(domain.type)
    velocity_sides = {
        side: [BoundaryConditionConfig(type="dirichlet", value=_vector_zero())]
        for side in domain_spec.boundary_roles["all"]
    }
    temperature_sides = {
        "x-": [BoundaryConditionConfig(type="neumann", value=_constant(0.0))],
        "x+": [BoundaryConditionConfig(type="neumann", value=_constant(0.0))],
        "y-": [BoundaryConditionConfig(type="dirichlet", value=_constant(0.5))],
        "y+": [BoundaryConditionConfig(type="dirichlet", value=_constant(-0.5))],
    }
    return {
        "velocity": BoundaryFieldConfig(sides=velocity_sides),
        "temperature": BoundaryFieldConfig(sides=temperature_sides),
    }


SCENARIO = register_boundary_scenario(
    BoundaryScenario(
        spec=BoundaryScenarioSpec(
            name="thermal_convection_rayleigh_benard",
            description="No-slip velocity with hot lower and cold upper walls.",
            supported_dimensions=(2,),
            supported_pdes=("thermal_convection",),
            supported_domains=("rectangle",),
            required_boundary_roles=("all",),
            required_boundary_names=("x-", "x+", "y-", "y+"),
            configured_fields=("velocity", "temperature"),
            field_shapes=("vector", "scalar"),
            operators=("dirichlet", "neumann"),
            level="pde",
        ),
        build=_build,
    )
)

"""Elasticity scenario with clamped horizontal edges."""

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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plm_data.sampling.context import SamplingContext


def _vector_zero() -> FieldExpressionConfig:
    return FieldExpressionConfig(type="zero")


def _build(
    context: "SamplingContext",
    domain: DomainConfig,
) -> dict[str, BoundaryFieldConfig]:
    if context.pde_name != "elasticity":
        raise ValueError(
            f"elasticity_clamped_y supports only elasticity. Got {context.pde_name!r}."
        )
    zero = _vector_zero()
    return {
        "displacement": BoundaryFieldConfig(
            sides={
                "x-": [BoundaryConditionConfig(type="neumann", value=zero)],
                "x+": [BoundaryConditionConfig(type="neumann", value=zero)],
                "y-": [BoundaryConditionConfig(type="dirichlet", value=zero)],
                "y+": [BoundaryConditionConfig(type="dirichlet", value=zero)],
            }
        )
    }


SCENARIO = register_boundary_scenario(
    BoundaryScenario(
        spec=BoundaryScenarioSpec(
            name="elasticity_clamped_y",
            description="Clamped y-min/y-max sides with free x-min/x-max sides.",
            supported_dimensions=(2,),
            supported_pdes=("elasticity",),
            supported_domains=("rectangle",),
            required_boundary_roles=("all",),
            required_boundary_names=("x-", "x+", "y-", "y+"),
            configured_fields=("displacement",),
            field_shapes=("vector",),
            operators=("dirichlet", "neumann"),
            level="pde",
        ),
        build=_build,
    )
)

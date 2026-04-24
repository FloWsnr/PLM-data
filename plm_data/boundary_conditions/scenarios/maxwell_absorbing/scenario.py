"""Maxwell absorbing boundary scenario."""

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


def _vector_zero() -> FieldExpressionConfig:
    return FieldExpressionConfig(type="zero")


def _build(
    context: "SamplingContext",
    domain: DomainConfig,
) -> dict[str, BoundaryFieldConfig]:
    if context.pde_name != "maxwell_pulse":
        raise ValueError(
            f"maxwell_absorbing supports only maxwell_pulse. Got {context.pde_name!r}."
        )
    domain_spec = get_domain_spec(domain.type)
    sides = {
        side: [BoundaryConditionConfig(type="absorbing", value=_vector_zero())]
        for side in domain_spec.boundary_roles["all"]
    }
    return {"electric_field": BoundaryFieldConfig(sides=sides)}


SCENARIO = register_boundary_scenario(
    BoundaryScenario(
        spec=BoundaryScenarioSpec(
            name="maxwell_absorbing",
            description="Homogeneous absorbing boundary on every side.",
            supported_dimensions=(2,),
            supported_pdes=("maxwell_pulse",),
            supported_domains=("rectangle",),
            required_boundary_roles=("all",),
            required_boundary_names=(),
            configured_fields=("electric_field",),
            field_shapes=("vector",),
            operators=("absorbing",),
            level="pde",
        ),
        build=_build,
    )
)

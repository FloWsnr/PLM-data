"""Simply supported plate boundary scenario."""

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
)
from plm_data.domains.registry import get_domain_spec

if TYPE_CHECKING:
    from plm_data.sampling.context import SamplingContext


def _build(
    context: "SamplingContext",
    domain: DomainConfig,
) -> dict[str, BoundaryFieldConfig]:
    if context.pde_name != "plate":
        raise ValueError(
            f"plate_simply_supported supports only plate. Got {context.pde_name!r}."
        )
    domain_spec = get_domain_spec(domain.type)
    sides = {
        side: [BoundaryConditionConfig(type="simply_supported")]
        for side in domain_spec.boundary_roles["all"]
    }
    return {"deflection": BoundaryFieldConfig(sides=sides)}


SCENARIO = register_boundary_scenario(
    BoundaryScenario(
        spec=BoundaryScenarioSpec(
            name="plate_simply_supported",
            description="Homogeneous simply-supported edges on every boundary.",
            supported_dimensions=(2,),
            supported_pdes=("plate",),
            supported_domains=("rectangle",),
            required_boundary_roles=("all",),
            required_boundary_names=(),
            configured_fields=("deflection",),
            field_shapes=("scalar",),
            operators=("simply_supported",),
            level="pde",
        ),
        build=_build,
    )
)

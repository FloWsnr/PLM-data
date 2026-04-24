"""Full-periodic shallow-water boundary scenario."""

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


def _periodic_sides(domain: DomainConfig) -> dict[str, list[BoundaryConditionConfig]]:
    domain_spec = get_domain_spec(domain.type)
    paired_sides: dict[str, str] = {}
    for side_a, side_b in domain_spec.periodic_pairs:
        paired_sides[side_a] = side_b
        paired_sides[side_b] = side_a
    missing = set(domain_spec.boundary_names) - set(paired_sides)
    if missing:
        raise ValueError(
            f"Domain '{domain.type}' does not expose full periodic pairs for "
            f"boundaries {sorted(missing)}."
        )
    return {
        side: [BoundaryConditionConfig(type="periodic", pair_with=pair_with)]
        for side, pair_with in paired_sides.items()
    }


def _build(
    context: "SamplingContext",
    domain: DomainConfig,
) -> dict[str, BoundaryFieldConfig]:
    if context.pde_name != "shallow_water":
        raise ValueError(
            "shallow_water_full_periodic supports only shallow_water. "
            f"Got {context.pde_name!r}."
        )
    sides = _periodic_sides(domain)
    return {
        "height": BoundaryFieldConfig(sides=sides.copy()),
        "velocity": BoundaryFieldConfig(sides=sides.copy()),
    }


SCENARIO = register_boundary_scenario(
    BoundaryScenario(
        spec=BoundaryScenarioSpec(
            name="shallow_water_full_periodic",
            description="Periodic height and velocity on all paired boundaries.",
            supported_dimensions=(2,),
            supported_pdes=("shallow_water",),
            supported_domains=("rectangle",),
            required_boundary_roles=(),
            required_boundary_names=(),
            configured_fields=("height", "velocity"),
            field_shapes=("scalar", "vector"),
            operators=("periodic",),
            level="pde",
        ),
        build=_build,
    )
)

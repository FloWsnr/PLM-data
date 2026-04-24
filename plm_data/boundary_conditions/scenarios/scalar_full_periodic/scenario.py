"""Scalar full-periodic boundary scenario."""

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

_SUPPORTED_PDES = (
    "cahn_hilliard",
    "cgl",
    "kuramoto_sivashinsky",
    "swift_hohenberg",
    "zakharov_kuznetsov",
)


def _build(
    context: "SamplingContext",
    domain: DomainConfig,
) -> dict[str, BoundaryFieldConfig]:
    if context.pde_name not in _SUPPORTED_PDES:
        raise ValueError(
            "scalar_full_periodic supports only migrated scalar periodic PDEs. "
            f"Got {context.pde_name!r}."
        )

    from plm_data.pdes import get_pde

    boundary_fields = get_pde(context.pde_name).spec.boundary_fields
    non_scalar_fields = [
        name
        for name, field_spec in boundary_fields.items()
        if field_spec.shape != "scalar"
    ]
    if non_scalar_fields:
        raise ValueError(
            "scalar_full_periodic can only configure scalar boundary fields. "
            f"Non-scalar fields: {non_scalar_fields}."
        )

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

    sides = {
        side: [BoundaryConditionConfig(type="periodic", pair_with=pair_with)]
        for side, pair_with in paired_sides.items()
    }
    return {
        field_name: BoundaryFieldConfig(sides=sides.copy())
        for field_name in boundary_fields
    }


SCENARIO = register_boundary_scenario(
    BoundaryScenario(
        spec=BoundaryScenarioSpec(
            name="scalar_full_periodic",
            description="Periodic constraints on every paired scalar boundary.",
            supported_dimensions=(2,),
            supported_pdes=_SUPPORTED_PDES,
            supported_domains=("rectangle",),
            required_boundary_roles=(),
            required_boundary_names=(),
            configured_fields=("all_scalar_boundary_fields",),
            field_shapes=("scalar",),
            operators=("periodic",),
            level="pde",
        ),
        build=_build,
    )
)

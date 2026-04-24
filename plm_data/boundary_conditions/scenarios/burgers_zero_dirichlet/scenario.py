"""Zero-Dirichlet Burgers boundary scenario."""

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
    if context.pde_name != "burgers":
        raise ValueError(
            f"burgers_zero_dirichlet supports only burgers. Got {context.pde_name!r}."
        )
    domain_spec = get_domain_spec(domain.type)
    zero = _vector_zero()
    sides = {
        side: [BoundaryConditionConfig(type="dirichlet", value=zero)]
        for side in domain_spec.boundary_roles["all"]
    }
    return {"velocity": BoundaryFieldConfig(sides=sides)}


SCENARIO = register_boundary_scenario(
    BoundaryScenario(
        spec=BoundaryScenarioSpec(
            name="burgers_zero_dirichlet",
            description="Homogeneous vector Dirichlet velocity on every boundary.",
            supported_dimensions=(2,),
            supported_pdes=("burgers",),
            supported_domains=("rectangle",),
            required_boundary_roles=("all",),
            required_boundary_names=(),
            configured_fields=("velocity",),
            field_shapes=("vector",),
            operators=("dirichlet",),
            level="pde",
        ),
        build=_build,
    )
)

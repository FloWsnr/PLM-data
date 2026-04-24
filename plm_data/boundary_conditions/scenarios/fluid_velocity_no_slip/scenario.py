"""No-slip velocity boundary scenario."""

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
    if context.pde_name != "navier_stokes":
        raise ValueError(
            "fluid_velocity_no_slip supports only navier_stokes. "
            f"Got {context.pde_name!r}."
        )
    domain_spec = get_domain_spec(domain.type)
    sides = {
        side: [BoundaryConditionConfig(type="dirichlet", value=_vector_zero())]
        for side in domain_spec.boundary_roles["all"]
    }
    return {"velocity": BoundaryFieldConfig(sides=sides)}


SCENARIO = register_boundary_scenario(
    BoundaryScenario(
        spec=BoundaryScenarioSpec(
            name="fluid_velocity_no_slip",
            description="Zero velocity on every domain boundary.",
            supported_dimensions=(2,),
            supported_pdes=("navier_stokes",),
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

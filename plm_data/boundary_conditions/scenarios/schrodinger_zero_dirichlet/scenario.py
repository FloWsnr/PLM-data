"""Zero-Dirichlet Schrodinger boundary scenario."""

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


def _zero() -> FieldExpressionConfig:
    return FieldExpressionConfig(type="constant", params={"value": 0.0})


def _build(
    context: "SamplingContext",
    domain: DomainConfig,
) -> dict[str, BoundaryFieldConfig]:
    if context.pde_name != "schrodinger":
        raise ValueError(
            "schrodinger_zero_dirichlet supports only schrodinger. "
            f"Got {context.pde_name!r}."
        )
    domain_spec = get_domain_spec(domain.type)
    zero = _zero()
    sides = {
        side: [BoundaryConditionConfig(type="dirichlet", value=zero)]
        for side in domain_spec.boundary_roles["all"]
    }
    return {
        "u": BoundaryFieldConfig(sides=sides),
        "v": BoundaryFieldConfig(sides=sides),
    }


SCENARIO = register_boundary_scenario(
    BoundaryScenario(
        spec=BoundaryScenarioSpec(
            name="schrodinger_zero_dirichlet",
            description="Homogeneous Dirichlet real/imaginary boundaries.",
            supported_dimensions=(2,),
            supported_pdes=("schrodinger",),
            supported_domains=("rectangle",),
            required_boundary_roles=("all",),
            required_boundary_names=(),
            configured_fields=("u", "v"),
            field_shapes=("scalar",),
            operators=("dirichlet",),
            level="pde",
        ),
        build=_build,
    )
)

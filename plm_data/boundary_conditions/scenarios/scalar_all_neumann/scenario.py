"""Scalar all-Neumann boundary scenario."""

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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plm_data.sampling.context import SamplingContext

_SUPPORTED_PDES = (
    "advection",
    "wave",
    "heat",
    "bistable_travelling_waves",
    "fisher_kpp",
    "brusselator",
    "fitzhugh_nagumo",
    "gray_scott",
    "gierer_meinhardt",
    "schnakenberg",
    "cyclic_competition",
    "immunotherapy",
    "van_der_pol",
    "lorenz",
    "keller_segel",
    "klausmeier_topography",
    "superlattice",
)


def _constant(value: float) -> FieldExpressionConfig:
    return FieldExpressionConfig(type="constant", params={"value": value})


def _build(
    context: "SamplingContext",
    domain: DomainConfig,
) -> dict[str, BoundaryFieldConfig]:
    if context.pde_name not in _SUPPORTED_PDES:
        raise ValueError(
            "scalar_all_neumann supports only scalar migrated PDEs. "
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
            "scalar_all_neumann can only configure scalar boundary fields. "
            f"Non-scalar fields: {non_scalar_fields}."
        )

    domain_spec = get_domain_spec(domain.type)
    sides = {
        side: [BoundaryConditionConfig(type="neumann", value=_constant(0.0))]
        for side in domain_spec.boundary_roles["all"]
    }
    return {
        field_name: BoundaryFieldConfig(sides=sides.copy())
        for field_name in boundary_fields
    }


SCENARIO = register_boundary_scenario(
    BoundaryScenario(
        spec=BoundaryScenarioSpec(
            name="scalar_all_neumann",
            description="Homogeneous Neumann conditions on every domain boundary.",
            supported_dimensions=(2,),
            supported_pdes=_SUPPORTED_PDES,
            supported_domains=("rectangle", "disk", "annulus"),
            required_boundary_roles=("all",),
            required_boundary_names=(),
            configured_fields=("all_scalar_boundary_fields",),
            field_shapes=("scalar",),
            operators=("neumann",),
            level="pde",
        ),
        build=_build,
    )
)

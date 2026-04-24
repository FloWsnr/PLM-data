"""Darcy pressure-drive boundary scenario."""

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


def _constant(value: float | str) -> FieldExpressionConfig:
    return FieldExpressionConfig(type="constant", params={"value": value})


def _build(
    context: "SamplingContext",
    domain: DomainConfig,
) -> dict[str, BoundaryFieldConfig]:
    if context.pde_name != "darcy":
        raise ValueError(
            f"darcy_pressure_drive supports only darcy. Got {context.pde_name!r}."
        )
    return {
        "pressure": BoundaryFieldConfig(
            sides={
                "x-": [
                    BoundaryConditionConfig(
                        type="dirichlet",
                        value=_constant("param:inlet_pressure"),
                    )
                ],
                "x+": [
                    BoundaryConditionConfig(
                        type="dirichlet",
                        value=_constant("param:outlet_pressure"),
                    )
                ],
                "y-": [BoundaryConditionConfig(type="neumann", value=_constant(0.0))],
                "y+": [BoundaryConditionConfig(type="neumann", value=_constant(0.0))],
            }
        ),
        "concentration": BoundaryFieldConfig(
            sides={
                "x-": [BoundaryConditionConfig(type="dirichlet", value=_constant(0.0))],
                "x+": [BoundaryConditionConfig(type="neumann", value=_constant(0.0))],
                "y-": [BoundaryConditionConfig(type="neumann", value=_constant(0.0))],
                "y+": [BoundaryConditionConfig(type="neumann", value=_constant(0.0))],
            }
        ),
    }


SCENARIO = register_boundary_scenario(
    BoundaryScenario(
        spec=BoundaryScenarioSpec(
            name="darcy_pressure_drive",
            description=(
                "Pressure Dirichlet drive from x-min to x-max with passive tracer "
                "inflow."
            ),
            supported_dimensions=(2,),
            supported_pdes=("darcy",),
            supported_domains=("rectangle",),
            required_boundary_roles=("all",),
            required_boundary_names=("x-", "x+", "y-", "y+"),
            configured_fields=("pressure", "concentration"),
            field_shapes=("scalar",),
            operators=("dirichlet", "neumann"),
            level="pde",
        ),
        build=_build,
    )
)

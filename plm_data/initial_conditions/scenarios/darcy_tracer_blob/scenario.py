"""Darcy tracer-blob initial-condition scenario."""

from typing import TYPE_CHECKING

from plm_data.core.runtime_config import (
    DomainConfig,
    FieldExpressionConfig,
    InputConfig,
)
from plm_data.initial_conditions.scenarios.base import (
    InitialConditionScenario,
    InitialConditionScenarioSpec,
    register_initial_condition_scenario,
)

if TYPE_CHECKING:
    from plm_data.sampling.context import SamplingContext


def _uniform(
    context: "SamplingContext",
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    value = float(context.child(name).rng.uniform(minimum, maximum))
    context.values[name] = value
    return value


def _expr(expr_type: str, **params) -> FieldExpressionConfig:
    return FieldExpressionConfig(type=expr_type, params=params)


def _constant(value: float) -> FieldExpressionConfig:
    return _expr("constant", value=value)


def _build(
    context: "SamplingContext",
    domain: DomainConfig,
    parameters: dict[str, float],
) -> dict[str, InputConfig]:
    length, height = domain.params["size"]
    return {
        "pressure": InputConfig(
            source=_expr("none"),
            initial_condition=_constant(0.0),
        ),
        "concentration": InputConfig(
            source=_expr("none"),
            initial_condition=_expr(
                "gaussian_bump",
                amplitude=_uniform(context, "darcy.ic.amplitude", 0.75, 1.1),
                sigma=_uniform(context, "darcy.ic.sigma", 0.04, 0.08),
                center=[
                    _uniform(context, "darcy.ic.center_x", 0.2 * length, 0.45 * length),
                    _uniform(
                        context, "darcy.ic.center_y", 0.25 * height, 0.75 * height
                    ),
                ],
            ),
        ),
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="darcy_tracer_blob",
            description="Zero pressure with an upstream passive tracer blob.",
            supported_dimensions=(2,),
            supported_pdes=("darcy",),
            supported_domains=("rectangle",),
            configured_inputs=("pressure", "concentration"),
            field_shapes=("scalar",),
            operators=("constant", "gaussian_bump", "none"),
            coordinate_regions=("interior", "left_half"),
        ),
        build=_build,
    )
)

"""Fisher-KPP step-front initial-condition scenario."""

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


def _build(
    context: "SamplingContext",
    domain: DomainConfig,
    parameters: dict[str, float],
) -> dict[str, InputConfig]:
    length = domain.params["size"][0]
    carrying_capacity = parameters["K"]
    return {
        "u": InputConfig(
            initial_condition=_expr(
                "step",
                value_left=carrying_capacity,
                value_right=0.02 * carrying_capacity,
                x_split=_uniform(
                    context,
                    "fisher.ic.x_split",
                    0.25 * length,
                    0.45 * length,
                ),
                axis=0,
            )
        )
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="fisher_step_front",
            description="Left-to-right logistic invasion step front.",
            supported_dimensions=(2,),
            supported_pdes=("fisher_kpp",),
            supported_domains=("rectangle",),
            configured_inputs=("u",),
            field_shapes=("scalar",),
            operators=("step",),
            coordinate_regions=("left_half", "right_half"),
        ),
        build=_build,
    )
)

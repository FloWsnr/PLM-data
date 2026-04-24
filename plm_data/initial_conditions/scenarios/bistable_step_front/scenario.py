"""Bistable step-front initial-condition scenario."""

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
    if domain.type != "rectangle":
        raise ValueError(
            "bistable_step_front currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    length = domain.params["size"][0]
    return {
        "u": InputConfig(
            initial_condition=_expr(
                "step",
                value_left=_uniform(context, "bistable.ic.high", 0.82, 0.98),
                value_right=_uniform(context, "bistable.ic.low", 0.02, 0.08),
                x_split=_uniform(
                    context,
                    "bistable.ic.x_split",
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
            name="bistable_step_front",
            description="Left-to-right bistable front step.",
            supported_dimensions=(2,),
            supported_pdes=("bistable_travelling_waves",),
            supported_domains=("rectangle",),
            configured_inputs=("u",),
            field_shapes=("scalar",),
            operators=("step",),
            coordinate_regions=("left_half", "right_half"),
        ),
        build=_build,
    )
)

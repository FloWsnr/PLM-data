"""Cahn-Hilliard noisy-mixture initial-condition scenario."""

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
            "cahn_hilliard_noise currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    return {
        "c": InputConfig(
            initial_condition=_expr(
                "gaussian_noise",
                mean=_uniform(context, "cahn.ic.mean", 0.42, 0.58),
                std=_uniform(context, "cahn.ic.std", 0.025, 0.07),
            )
        )
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="cahn_hilliard_noise",
            description="Noisy concentration field around a mixed state.",
            supported_dimensions=(2,),
            supported_pdes=("cahn_hilliard",),
            supported_domains=("rectangle",),
            configured_inputs=("c",),
            field_shapes=("scalar",),
            operators=("gaussian_noise",),
            coordinate_regions=("interior",),
        ),
        build=_build,
    )
)

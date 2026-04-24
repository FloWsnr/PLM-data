"""Gierer-Meinhardt perturbed initial-condition scenario."""

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
            "gierer_meinhardt_perturbed currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    noise = _uniform(context, "gierer.ic.noise", 0.006, 0.018)
    return {
        "a": InputConfig(
            initial_condition=_expr(
                "gaussian_noise",
                mean=_uniform(context, "gierer.ic.a_mean", 0.35, 0.55),
                std=noise,
            )
        ),
        "h": InputConfig(
            initial_condition=_expr(
                "gaussian_noise",
                mean=_uniform(context, "gierer.ic.h_mean", 0.9, 1.15),
                std=noise,
            )
        ),
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="gierer_meinhardt_perturbed",
            description="Noisy positive activator/inhibitor initial fields.",
            supported_dimensions=(2,),
            supported_pdes=("gierer_meinhardt",),
            supported_domains=("rectangle",),
            configured_inputs=("a", "h"),
            field_shapes=("scalar",),
            operators=("gaussian_noise",),
            coordinate_regions=("interior",),
        ),
        build=_build,
    )
)

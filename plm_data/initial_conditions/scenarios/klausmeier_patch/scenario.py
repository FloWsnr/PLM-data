"""Klausmeier patch initial-condition scenario."""

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
            "klausmeier_patch currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    return {
        "w": InputConfig(
            initial_condition=_expr(
                "gaussian_noise",
                mean=_uniform(context, "klausmeier.ic.w_mean", 0.7, 1.05),
                std=_uniform(context, "klausmeier.ic.w_std", 0.015, 0.04),
            )
        ),
        "n": InputConfig(
            initial_condition=_expr(
                "gaussian_noise",
                mean=_uniform(context, "klausmeier.ic.n_mean", 0.12, 0.26),
                std=_uniform(context, "klausmeier.ic.n_std", 0.01, 0.035),
            )
        ),
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="klausmeier_patch",
            description="Noisy water and vegetation biomass fields.",
            supported_dimensions=(2,),
            supported_pdes=("klausmeier_topography",),
            supported_domains=("rectangle",),
            configured_inputs=("w", "n"),
            field_shapes=("scalar",),
            operators=("gaussian_noise",),
            coordinate_regions=("interior",),
        ),
        build=_build,
    )
)

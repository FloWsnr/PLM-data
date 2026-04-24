"""Lorenz noisy-state initial-condition scenario."""

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
            "lorenz_noisy_state currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    return {
        "x": InputConfig(
            initial_condition=_expr(
                "gaussian_noise",
                mean=_uniform(context, "lorenz.ic.x_mean", -0.6, 0.8),
                std=_uniform(context, "lorenz.ic.x_std", 1.2, 2.4),
            )
        ),
        "y": InputConfig(
            initial_condition=_expr(
                "gaussian_noise",
                mean=_uniform(context, "lorenz.ic.y_mean", -0.8, 0.6),
                std=_uniform(context, "lorenz.ic.y_std", 1.2, 2.6),
            )
        ),
        "z": InputConfig(
            initial_condition=_expr(
                "gaussian_noise",
                mean=_uniform(context, "lorenz.ic.z_mean", 24.0, 31.0),
                std=_uniform(context, "lorenz.ic.z_std", 1.6, 3.8),
            )
        ),
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="lorenz_noisy_state",
            description="Noisy Lorenz fields near the classical chaotic regime.",
            supported_dimensions=(2,),
            supported_pdes=("lorenz",),
            supported_domains=("rectangle",),
            configured_inputs=("x", "y", "z"),
            field_shapes=("scalar",),
            operators=("gaussian_noise",),
            coordinate_regions=("interior",),
        ),
        build=_build,
    )
)

"""Keller-Segel cluster initial-condition scenario."""

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
            "keller_segel_cluster currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    return {
        "rho": InputConfig(
            initial_condition=_expr(
                "gaussian_noise",
                mean=_uniform(context, "keller.ic.rho_mean", 0.35, 0.55),
                std=_uniform(context, "keller.ic.rho_std", 0.015, 0.04),
            )
        ),
        "c": InputConfig(
            initial_condition=_expr(
                "gaussian_noise",
                mean=_uniform(context, "keller.ic.c_mean", 0.18, 0.34),
                std=_uniform(context, "keller.ic.c_std", 0.01, 0.03),
            )
        ),
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="keller_segel_cluster",
            description="Noisy positive cell and chemoattractant fields.",
            supported_dimensions=(2,),
            supported_pdes=("keller_segel",),
            supported_domains=("rectangle",),
            configured_inputs=("rho", "c"),
            field_shapes=("scalar",),
            operators=("gaussian_noise",),
            coordinate_regions=("interior",),
        ),
        build=_build,
    )
)

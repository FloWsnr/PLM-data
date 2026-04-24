"""Immunotherapy patch initial-condition scenario."""

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
            "immunotherapy_patch currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    return {
        "u": InputConfig(
            initial_condition=_expr(
                "gaussian_noise",
                mean=_uniform(context, "immunotherapy.ic.u_mean", 0.09, 0.16),
                std=_uniform(context, "immunotherapy.ic.u_std", 0.006, 0.018),
            )
        ),
        "v": InputConfig(
            initial_condition=_expr(
                "gaussian_noise",
                mean=_uniform(context, "immunotherapy.ic.v_mean", 0.78, 0.9),
                std=_uniform(context, "immunotherapy.ic.v_std", 0.015, 0.04),
            )
        ),
        "w": InputConfig(
            initial_condition=_expr(
                "gaussian_noise",
                mean=_uniform(context, "immunotherapy.ic.w_mean", 0.012, 0.024),
                std=_uniform(context, "immunotherapy.ic.w_std", 0.001, 0.0035),
            )
        ),
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="immunotherapy_patch",
            description="Noisy tumour/immune/cytokine patch state.",
            supported_dimensions=(2,),
            supported_pdes=("immunotherapy",),
            supported_domains=("rectangle",),
            configured_inputs=("u", "v", "w"),
            field_shapes=("scalar",),
            operators=("gaussian_noise",),
            coordinate_regions=("interior",),
        ),
        build=_build,
    )
)

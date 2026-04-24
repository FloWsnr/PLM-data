"""Complex Ginzburg-Landau noisy-wave initial-condition scenario."""

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
            "cgl_noisy_wave currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    amplitude = _uniform(context, "cgl.ic.amplitude", 0.18, 0.34)
    noise = _uniform(context, "cgl.ic.noise", 0.008, 0.02)
    return {
        "u": InputConfig(
            initial_condition=_expr("gaussian_noise", mean=amplitude, std=noise)
        ),
        "v": InputConfig(
            initial_condition=_expr(
                "gaussian_noise",
                mean=_uniform(context, "cgl.ic.v_mean", -0.05, 0.05),
                std=noise,
            )
        ),
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="cgl_noisy_wave",
            description="Small noisy complex-amplitude perturbation.",
            supported_dimensions=(2,),
            supported_pdes=("cgl",),
            supported_domains=("rectangle",),
            configured_inputs=("u", "v"),
            field_shapes=("scalar",),
            operators=("gaussian_noise",),
            coordinate_regions=("interior",),
        ),
        build=_build,
    )
)

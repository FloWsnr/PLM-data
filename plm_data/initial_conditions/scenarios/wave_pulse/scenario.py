"""Wave pulse initial-condition scenario."""

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
            "wave_pulse currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    length, height = domain.params["size"]
    scale = min(float(length), float(height))
    center = [
        _uniform(context, "wave.ic.center_x", 0.35 * length, 0.65 * length),
        _uniform(context, "wave.ic.center_y", 0.35 * height, 0.65 * height),
    ]
    return {
        "u": InputConfig(
            initial_condition=_expr(
                "gaussian_bump",
                amplitude=_uniform(context, "wave.ic.amplitude", 0.7, 1.1),
                sigma=_uniform(context, "wave.ic.sigma", 0.06 * scale, 0.11 * scale),
                center=center,
            ),
        ),
        "v": InputConfig(initial_condition=_expr("constant", value=0.0)),
        "forcing": InputConfig(source=_expr("none")),
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="wave_pulse",
            description="Centered scalar displacement pulse with zero initial velocity.",
            supported_dimensions=(2,),
            supported_pdes=("wave",),
            supported_domains=("rectangle",),
            configured_inputs=("u", "v", "forcing"),
            field_shapes=("scalar",),
            operators=("gaussian_bump", "constant", "none"),
            coordinate_regions=("interior", "center"),
        ),
        build=_build,
    )
)

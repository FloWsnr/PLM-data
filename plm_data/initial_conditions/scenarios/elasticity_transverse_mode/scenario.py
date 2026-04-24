"""Elasticity transverse-mode initial-condition scenario."""

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


def _constant(value: float) -> FieldExpressionConfig:
    return _expr("constant", value=value)


def _vector_zero() -> FieldExpressionConfig:
    return FieldExpressionConfig(type="zero")


def _vector_components(
    x: FieldExpressionConfig,
    y: FieldExpressionConfig,
) -> FieldExpressionConfig:
    return FieldExpressionConfig(components={"x": x, "y": y})


def _build(
    context: "SamplingContext",
    domain: DomainConfig,
    parameters: dict[str, float],
) -> dict[str, InputConfig]:
    length, height = domain.params["size"]
    displacement_ic = _vector_components(
        _constant(0.0),
        _expr(
            "sine_waves",
            background=0.0,
            modes=[
                {
                    "amplitude": _uniform(context, "elasticity.ic.amp", 0.025, 0.05),
                    "cycles": [1.0 / length, 1.0 / height],
                    "phase": _uniform(context, "elasticity.ic.phase", -0.5, 0.5),
                    "angle": 0.0,
                }
            ],
        ),
    )
    return {
        "displacement": InputConfig(initial_condition=displacement_ic),
        "velocity": InputConfig(initial_condition=_vector_zero()),
        "forcing": InputConfig(source=_vector_zero()),
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="elasticity_transverse_mode",
            description="Small transverse displacement mode with zero velocity.",
            supported_dimensions=(2,),
            supported_pdes=("elasticity",),
            supported_domains=("rectangle",),
            configured_inputs=("displacement", "velocity", "forcing"),
            field_shapes=("vector",),
            operators=("constant", "sine_waves", "zero"),
            coordinate_regions=("interior",),
        ),
        build=_build,
    )
)

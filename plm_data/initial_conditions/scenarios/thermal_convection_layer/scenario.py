"""Thermal-convection layer initial-condition scenario."""

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

_TWO_PI = 6.283185307179586


def _uniform(
    context: "SamplingContext",
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    value = float(context.child(name).rng.uniform(minimum, maximum))
    context.values[name] = value
    return value


def _build(
    context: "SamplingContext",
    domain: DomainConfig,
    parameters: dict[str, float],
) -> dict[str, InputConfig]:
    if domain.type != "rectangle":
        raise ValueError(
            "thermal_convection_layer supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    length, height = domain.params["size"]
    return {
        "velocity": InputConfig(
            source=FieldExpressionConfig(type="zero"),
            initial_condition=FieldExpressionConfig(type="zero"),
        ),
        "temperature": InputConfig(
            source=FieldExpressionConfig(type="zero"),
            initial_condition=FieldExpressionConfig(
                type="sine_waves",
                params={
                    "background": 0.0,
                    "modes": [
                        {
                            "amplitude": _uniform(
                                context,
                                "thermal.ic.amp0",
                                0.08,
                                0.16,
                            ),
                            "cycles": [1.0 / length, 1.0 / height],
                            "phase": _uniform(
                                context,
                                "thermal.ic.phase0",
                                0.0,
                                _TWO_PI,
                            ),
                            "angle": 0.0,
                        },
                        {
                            "amplitude": _uniform(
                                context,
                                "thermal.ic.amp1",
                                -0.05,
                                0.05,
                            ),
                            "cycles": [2.0 / length, 1.0 / height],
                            "phase": _uniform(
                                context,
                                "thermal.ic.phase1",
                                0.0,
                                _TWO_PI,
                            ),
                            "angle": 0.0,
                        },
                    ],
                },
            ),
        ),
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="thermal_convection_layer",
            description="Linear hot-bottom/cold-top temperature layer at rest.",
            supported_dimensions=(2,),
            supported_pdes=("thermal_convection",),
            supported_domains=("rectangle",),
            configured_inputs=("velocity", "temperature"),
            field_shapes=("vector", "scalar"),
            operators=("zero", "sine_waves"),
            coordinate_regions=("interior",),
        ),
        build=_build,
    )
)

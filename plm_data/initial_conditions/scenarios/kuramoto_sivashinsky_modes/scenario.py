"""Kuramoto-Sivashinsky low-mode initial-condition scenario."""

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


def _expr(expr_type: str, **params) -> FieldExpressionConfig:
    return FieldExpressionConfig(type=expr_type, params=params)


def _build(
    context: "SamplingContext",
    domain: DomainConfig,
    parameters: dict[str, float],
) -> dict[str, InputConfig]:
    if domain.type != "rectangle":
        raise ValueError(
            "kuramoto_sivashinsky_modes currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    return {
        "u": InputConfig(
            initial_condition=_expr(
                "sine_waves",
                background=0.0,
                modes=[
                    {
                        "amplitude": _uniform(context, "ks.ic.amp0", 0.03, 0.08),
                        "cycles": [1.0, 1.0],
                        "phase": _uniform(context, "ks.ic.phase0", 0.0, _TWO_PI),
                    },
                    {
                        "amplitude": _uniform(context, "ks.ic.amp1", -0.05, 0.05),
                        "cycles": [2.0, 1.0],
                        "phase": _uniform(context, "ks.ic.phase1", 0.0, _TWO_PI),
                    },
                ],
            )
        )
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="kuramoto_sivashinsky_modes",
            description="Low-amplitude modal perturbation for chaotic onset.",
            supported_dimensions=(2,),
            supported_pdes=("kuramoto_sivashinsky",),
            supported_domains=("rectangle",),
            configured_inputs=("u",),
            field_shapes=("scalar",),
            operators=("sine_waves",),
            coordinate_regions=("interior",),
        ),
        build=_build,
    )
)

"""Schrodinger Gaussian wave-packet initial-condition scenario."""

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

_HALF_PI = 1.5707963267948966


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
            "schrodinger_wave_packet currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    length, height = domain.params["size"]
    scale = min(float(length), float(height))
    center = [
        _uniform(context, "schrodinger.ic.center_x", 0.35 * length, 0.65 * length),
        _uniform(context, "schrodinger.ic.center_y", 0.35 * height, 0.65 * height),
    ]
    wavevector = [
        _uniform(context, "schrodinger.ic.wavevector_x", 5.0, 9.0),
        _uniform(context, "schrodinger.ic.wavevector_y", -2.5, 2.5),
    ]
    amplitude = _uniform(context, "schrodinger.ic.amplitude", 0.7, 1.0)
    sigma = _uniform(context, "schrodinger.ic.sigma", 0.08 * scale, 0.14 * scale)
    return {
        "u": InputConfig(
            initial_condition=_expr(
                "gaussian_wave_packet",
                amplitude=amplitude,
                sigma=sigma,
                center=center,
                wavevector=wavevector,
                phase=0.0,
            ),
        ),
        "v": InputConfig(
            initial_condition=_expr(
                "gaussian_wave_packet",
                amplitude=amplitude,
                sigma=sigma,
                center=center,
                wavevector=wavevector,
                phase=_HALF_PI,
            ),
        ),
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="schrodinger_wave_packet",
            description="Complex Gaussian wave packet split into real/imaginary fields.",
            supported_dimensions=(2,),
            supported_pdes=("schrodinger",),
            supported_domains=("rectangle",),
            configured_inputs=("u", "v"),
            field_shapes=("scalar",),
            operators=("gaussian_wave_packet",),
            coordinate_regions=("interior", "center"),
        ),
        build=_build,
    )
)

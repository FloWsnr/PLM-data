"""Burgers velocity-bump initial-condition scenario."""

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


def _vector_expr(**components: FieldExpressionConfig) -> FieldExpressionConfig:
    return FieldExpressionConfig(components=components)


def _build(
    context: "SamplingContext",
    domain: DomainConfig,
    parameters: dict[str, float],
) -> dict[str, InputConfig]:
    if domain.type != "rectangle":
        raise ValueError(
            "burgers_velocity_bump currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    length, height = domain.params["size"]
    scale = min(float(length), float(height))
    center = [
        _uniform(context, "burgers.ic.center_x", 0.35 * length, 0.65 * length),
        _uniform(context, "burgers.ic.center_y", 0.35 * height, 0.65 * height),
    ]
    sigma = _uniform(context, "burgers.ic.sigma", 0.09 * scale, 0.16 * scale)
    x_amplitude = _uniform(context, "burgers.ic.x_amplitude", 0.35, 0.65)
    y_amplitude = _uniform(context, "burgers.ic.y_amplitude", -0.35, 0.35)
    return {
        "velocity": InputConfig(
            source=FieldExpressionConfig(type="zero"),
            initial_condition=_vector_expr(
                x=_expr(
                    "gaussian_bump",
                    amplitude=x_amplitude,
                    sigma=sigma,
                    center=center,
                ),
                y=_expr(
                    "gaussian_bump",
                    amplitude=y_amplitude,
                    sigma=sigma,
                    center=center,
                ),
            ),
        )
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="burgers_velocity_bump",
            description="Centered vector Gaussian velocity impulse with no forcing.",
            supported_dimensions=(2,),
            supported_pdes=("burgers",),
            supported_domains=("rectangle",),
            configured_inputs=("velocity",),
            field_shapes=("vector",),
            operators=("gaussian_bump", "zero"),
            coordinate_regions=("interior", "center"),
        ),
        build=_build,
    )
)

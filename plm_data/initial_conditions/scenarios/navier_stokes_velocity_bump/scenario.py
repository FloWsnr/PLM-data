"""Navier-Stokes localized velocity initial-condition scenario."""

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
            "navier_stokes_velocity_bump currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    length, height = domain.params["size"]
    scale = min(float(length), float(height))
    center = [
        _uniform(context, "navier.ic.center_x", 0.35 * length, 0.65 * length),
        _uniform(context, "navier.ic.center_y", 0.35 * height, 0.65 * height),
    ]
    sigma = _uniform(context, "navier.ic.sigma", 0.08 * scale, 0.14 * scale)
    return {
        "velocity": InputConfig(
            source=FieldExpressionConfig(type="zero"),
            initial_condition=_vector_expr(
                x=_expr(
                    "gaussian_bump",
                    amplitude=_uniform(context, "navier.ic.x_amp", 0.04, 0.1),
                    sigma=sigma,
                    center=center,
                ),
                y=_expr(
                    "gaussian_bump",
                    amplitude=_uniform(context, "navier.ic.y_amp", -0.05, 0.05),
                    sigma=sigma,
                    center=center,
                ),
            ),
        )
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="navier_stokes_velocity_bump",
            description="Small localized velocity perturbation with no body force.",
            supported_dimensions=(2,),
            supported_pdes=("navier_stokes",),
            supported_domains=("rectangle",),
            configured_inputs=("velocity",),
            field_shapes=("vector",),
            operators=("gaussian_bump", "zero"),
            coordinate_regions=("interior",),
        ),
        build=_build,
    )
)

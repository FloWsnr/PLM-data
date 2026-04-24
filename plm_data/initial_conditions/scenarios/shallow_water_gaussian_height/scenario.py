"""Shallow-water Gaussian-height initial-condition scenario."""

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
            "shallow_water_gaussian_height supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    length, height = domain.params["size"]
    scale = min(float(length), float(height))
    return {
        "height": InputConfig(
            initial_condition=_expr(
                "gaussian_bump",
                amplitude=_uniform(context, "shallow.ic.height_amp", 0.015, 0.04),
                sigma=_uniform(
                    context,
                    "shallow.ic.sigma",
                    0.09 * scale,
                    0.16 * scale,
                ),
                center=[
                    _uniform(
                        context, "shallow.ic.center_x", 0.35 * length, 0.65 * length
                    ),
                    _uniform(
                        context, "shallow.ic.center_y", 0.35 * height, 0.65 * height
                    ),
                ],
            )
        ),
        "velocity": InputConfig(initial_condition=FieldExpressionConfig(type="zero")),
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="shallow_water_gaussian_height",
            description="Small localized free-surface height anomaly at rest.",
            supported_dimensions=(2,),
            supported_pdes=("shallow_water",),
            supported_domains=("rectangle",),
            configured_inputs=("height", "velocity"),
            field_shapes=("scalar", "vector"),
            operators=("gaussian_bump", "zero"),
            coordinate_regions=("interior", "center"),
        ),
        build=_build,
    )
)

"""Advection Gaussian-bump initial-condition scenario."""

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
            "advection_gaussian_bump currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    length, height = domain.params["size"]
    scale = min(float(length), float(height))
    center = [
        _uniform(context, "advection.ic.center_x", 0.25 * length, 0.45 * length),
        _uniform(context, "advection.ic.center_y", 0.3 * height, 0.7 * height),
    ]
    return {
        "u": InputConfig(
            source=_expr("none"),
            initial_condition=_expr(
                "gaussian_bump",
                amplitude=_uniform(context, "advection.ic.amplitude", 0.8, 1.2),
                sigma=_uniform(
                    context,
                    "advection.ic.sigma",
                    0.08 * scale,
                    0.14 * scale,
                ),
                center=center,
            ),
        )
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="advection_gaussian_bump",
            description="One upstream scalar Gaussian bump with no source.",
            supported_dimensions=(2,),
            supported_pdes=("advection",),
            supported_domains=("rectangle",),
            configured_inputs=("u",),
            field_shapes=("scalar",),
            operators=("gaussian_bump", "none"),
            coordinate_regions=("left_half", "interior"),
        ),
        build=_build,
    )
)

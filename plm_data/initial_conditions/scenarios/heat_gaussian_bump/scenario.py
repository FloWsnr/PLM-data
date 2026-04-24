"""Heat Gaussian bump initial-condition scenario."""

from typing import TYPE_CHECKING

from plm_data.core.runtime_config import (
    DomainConfig,
    FieldExpressionConfig,
    InputConfig,
)
from plm_data.domains import sample_coordinate_region
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
    coordinate_sample = sample_coordinate_region(domain, "interior", context)
    return {
        "u": InputConfig(
            source=_expr("none"),
            initial_condition=_expr(
                "gaussian_bump",
                amplitude=_uniform(context, "heat.ic.amplitude", 0.8, 1.2),
                sigma=_uniform(
                    context,
                    "heat.ic.sigma",
                    0.07 * coordinate_sample.scale,
                    0.13 * coordinate_sample.scale,
                ),
                center=coordinate_sample.point,
            ),
        )
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="heat_gaussian_bump",
            description="One interior scalar Gaussian bump with no source.",
            supported_dimensions=(2,),
            supported_pdes=("heat",),
            supported_domains=("rectangle", "disk", "annulus"),
            configured_inputs=("u",),
            field_shapes=("scalar",),
            operators=("gaussian_bump", "none"),
            coordinate_regions=("interior",),
        ),
        build=_build,
    )
)

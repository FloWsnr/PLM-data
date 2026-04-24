"""FitzHugh-Nagumo pulse initial-condition scenario."""

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
            "fitzhugh_nagumo_pulse currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    length, height = domain.params["size"]
    scale = min(float(length), float(height))
    center = [
        _uniform(context, "fitzhugh.ic.center_x", 0.35 * length, 0.65 * length),
        _uniform(context, "fitzhugh.ic.center_y", 0.35 * height, 0.65 * height),
    ]
    return {
        "u": InputConfig(
            initial_condition=_expr(
                "gaussian_bump",
                amplitude=_uniform(context, "fitzhugh.ic.amplitude", 0.65, 0.95),
                sigma=_uniform(
                    context,
                    "fitzhugh.ic.sigma",
                    0.08 * scale,
                    0.14 * scale,
                ),
                center=center,
            ),
        ),
        "v": InputConfig(initial_condition=_expr("constant", value=0.0)),
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="fitzhugh_nagumo_pulse",
            description="Localized activator pulse with resting inhibitor.",
            supported_dimensions=(2,),
            supported_pdes=("fitzhugh_nagumo",),
            supported_domains=("rectangle",),
            configured_inputs=("u", "v"),
            field_shapes=("scalar",),
            operators=("gaussian_bump", "constant"),
            coordinate_regions=("interior", "center"),
        ),
        build=_build,
    )
)

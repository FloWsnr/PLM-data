"""Gray-Scott spot initial-condition scenario."""

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
            "gray_scott_spot currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    length, height = domain.params["size"]
    scale = min(float(length), float(height))
    center = [
        _uniform(context, "gray_scott.ic.center_x", 0.4 * length, 0.6 * length),
        _uniform(context, "gray_scott.ic.center_y", 0.4 * height, 0.6 * height),
    ]
    return {
        "u": InputConfig(initial_condition=_expr("constant", value=1.0)),
        "v": InputConfig(
            initial_condition=_expr(
                "gaussian_bump",
                amplitude=_uniform(context, "gray_scott.ic.amplitude", 0.25, 0.45),
                sigma=_uniform(
                    context,
                    "gray_scott.ic.sigma",
                    0.08 * scale,
                    0.14 * scale,
                ),
                center=center,
            ),
        ),
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="gray_scott_spot",
            description="Uniform substrate with one autocatalyst spot.",
            supported_dimensions=(2,),
            supported_pdes=("gray_scott",),
            supported_domains=("rectangle",),
            configured_inputs=("u", "v"),
            field_shapes=("scalar",),
            operators=("constant", "gaussian_bump"),
            coordinate_regions=("interior", "center"),
        ),
        build=_build,
    )
)

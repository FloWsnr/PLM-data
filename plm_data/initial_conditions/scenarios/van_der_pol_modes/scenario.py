"""Van der Pol modal initial-condition scenario."""

import math
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


def _mode(
    context: "SamplingContext",
    prefix: str,
    length: float,
    height: float,
    amplitude_range: tuple[float, float],
) -> dict:
    return {
        "amplitude": _uniform(context, f"{prefix}.amplitude", *amplitude_range),
        "cycles": [
            _uniform(context, f"{prefix}.cycles_x", 0.8 / length, 1.8 / length),
            _uniform(context, f"{prefix}.cycles_y", 0.6 / height, 1.6 / height),
        ],
        "phase": _uniform(context, f"{prefix}.phase", 0.0, 2.0 * math.pi),
        "angle": _uniform(context, f"{prefix}.angle", -0.7, 0.7),
    }


def _build(
    context: "SamplingContext",
    domain: DomainConfig,
    parameters: dict[str, float],
) -> dict[str, InputConfig]:
    if domain.type != "rectangle":
        raise ValueError(
            "van_der_pol_modes currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    length, height = domain.params["size"]
    return {
        "u": InputConfig(
            initial_condition=_expr(
                "sine_waves",
                background=_uniform(context, "vdp.ic.u_background", -0.08, 0.08),
                modes=[
                    _mode(context, "vdp.ic.u_mode0", length, height, (0.25, 0.55)),
                    _mode(context, "vdp.ic.u_mode1", length, height, (-0.35, -0.12)),
                ],
            )
        ),
        "v": InputConfig(
            initial_condition=_expr(
                "sine_waves",
                background=_uniform(context, "vdp.ic.v_background", -0.04, 0.04),
                modes=[
                    _mode(context, "vdp.ic.v_mode0", length, height, (0.16, 0.36)),
                    _mode(context, "vdp.ic.v_mode1", length, height, (-0.24, -0.06)),
                ],
            )
        ),
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="van_der_pol_modes",
            description="Low-frequency displacement and velocity sine modes.",
            supported_dimensions=(2,),
            supported_pdes=("van_der_pol",),
            supported_domains=("rectangle",),
            configured_inputs=("u", "v"),
            field_shapes=("scalar",),
            operators=("sine_waves",),
            coordinate_regions=("interior",),
        ),
        build=_build,
    )
)

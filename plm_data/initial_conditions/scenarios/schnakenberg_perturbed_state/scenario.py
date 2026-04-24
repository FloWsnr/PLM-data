"""Schnakenberg perturbed steady-state initial-condition scenario."""

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
            "schnakenberg_perturbed_state currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    u_equilibrium = parameters["a"] + parameters["b"]
    v_equilibrium = parameters["b"] / (u_equilibrium * u_equilibrium)
    noise = _uniform(context, "schnakenberg.ic.noise", 0.006, 0.02)
    return {
        "u": InputConfig(
            initial_condition=_expr(
                "gaussian_noise",
                mean=u_equilibrium,
                std=noise,
            )
        ),
        "v": InputConfig(
            initial_condition=_expr(
                "gaussian_noise",
                mean=v_equilibrium,
                std=noise,
            )
        ),
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="schnakenberg_perturbed_state",
            description="Noisy perturbation of the homogeneous Schnakenberg state.",
            supported_dimensions=(2,),
            supported_pdes=("schnakenberg",),
            supported_domains=("rectangle",),
            configured_inputs=("u", "v"),
            field_shapes=("scalar",),
            operators=("gaussian_noise",),
            coordinate_regions=("interior",),
        ),
        build=_build,
    )
)

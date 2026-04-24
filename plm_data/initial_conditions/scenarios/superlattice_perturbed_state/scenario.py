"""Superlattice perturbed-state initial-condition scenario."""

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


def _input(mean: float, std: float) -> InputConfig:
    return InputConfig(
        initial_condition=_expr(
            "gaussian_noise",
            mean=mean,
            std=std,
        )
    )


def _build(
    context: "SamplingContext",
    domain: DomainConfig,
    parameters: dict[str, float],
) -> dict[str, InputConfig]:
    if domain.type != "rectangle":
        raise ValueError(
            "superlattice_perturbed_state currently supports only rectangle domains. "
            f"Got {domain.type!r}."
        )

    noise = _uniform(context, "superlattice.ic.noise", 0.006, 0.018)
    u_1 = parameters["a"]
    v_1 = parameters["b"] / parameters["a"]
    u_2 = parameters["c"] / 5.0
    v_2 = 1.0 + u_2 * u_2
    return {
        "u_1": _input(u_1, noise),
        "v_1": _input(v_1, noise),
        "u_2": _input(u_2, noise),
        "v_2": _input(v_2, noise),
    }


SCENARIO = register_initial_condition_scenario(
    InitialConditionScenario(
        spec=InitialConditionScenarioSpec(
            name="superlattice_perturbed_state",
            description="Noisy perturbation of uncoupled superlattice equilibria.",
            supported_dimensions=(2,),
            supported_pdes=("superlattice",),
            supported_domains=("rectangle",),
            configured_inputs=("u_1", "v_1", "u_2", "v_2"),
            field_shapes=("scalar",),
            operators=("gaussian_noise",),
            coordinate_regions=("interior",),
        ),
        build=_build,
    )
)

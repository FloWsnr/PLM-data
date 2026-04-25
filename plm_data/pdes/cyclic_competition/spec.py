"""Cyclic competition PDE spec."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    SATURATING_STOCHASTIC_COUPLINGS,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
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


def _cyclic_base_diffusion(context: "SamplingContext") -> float:
    key = "cyclic_competition.diffusion"
    if key not in context.values:
        context.values[key] = _uniform(context, key, 0.006, 0.016)
    return float(context.values[key])


def _cyclic_diffusion_sampler(
    scale_min: float,
    scale_max: float,
) -> Callable[["SamplingContext", str], float]:
    def _sample(context: "SamplingContext", stream: str) -> float:
        scale = _uniform(context, f"{stream}.scale", scale_min, scale_max)
        return _cyclic_base_diffusion(context) * scale

    return _sample


PDE_SPEC = PDESpec(
    name="cyclic_competition",
    category="biology",
    description=(
        "Three-species competitive Lotka-Volterra reaction-diffusion system "
        "with cyclic dominance."
    ),
    equations={
        "u": "du/dt = Du * laplacian(u) + u * (1 - u - a * v - b * w)",
        "v": "dv/dt = Dv * laplacian(v) + v * (1 - b * u - v - a * w)",
        "w": "dw/dt = Dw * laplacian(w) + w * (1 - a * u - b * v - w)",
    },
    parameters=[
        PDEParameter(
            "Du",
            "Diffusion coefficient of species u",
            hard_min=0.0,
            sampling_min=0.0048,
            sampling_max=0.0192,
            sampler=_cyclic_diffusion_sampler(0.8, 1.2),
        ),
        PDEParameter(
            "Dv",
            "Diffusion coefficient of species v",
            hard_min=0.0,
            sampling_min=0.0036,
            sampling_max=0.0224,
            sampler=_cyclic_diffusion_sampler(0.6, 1.4),
        ),
        PDEParameter(
            "Dw",
            "Diffusion coefficient of species w",
            hard_min=0.0,
            sampling_min=0.0042,
            sampling_max=0.024,
            sampler=_cyclic_diffusion_sampler(0.7, 1.5),
        ),
        PDEParameter(
            "a",
            "Weak interspecific competition coefficient",
            hard_min=0.0,
            sampling_min=0.35,
            sampling_max=0.75,
        ),
        PDEParameter(
            "b",
            "Strong interspecific competition coefficient",
            hard_min=1.0,
            sampling_min=1.15,
            sampling_max=1.65,
        ),
    ],
    inputs={
        "u": InputSpec(
            name="u",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "v": InputSpec(
            name="v",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "w": InputSpec(
            name="w",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "u": BoundaryFieldSpec(
            name="u",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for species u.",
        ),
        "v": BoundaryFieldSpec(
            name="v",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for species v.",
        ),
        "w": BoundaryFieldSpec(
            name="w",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for species w.",
        ),
    },
    states={
        "u": StateSpec(
            name="u",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
        "v": StateSpec(
            name="v",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
        "w": StateSpec(
            name="w",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
    },
    outputs={
        "u": OutputSpec(
            name="u",
            shape="scalar",
            output_mode="scalar",
            source_name="u",
        ),
        "v": OutputSpec(
            name="v",
            shape="scalar",
            output_mode="scalar",
            source_name="v",
        ),
        "w": OutputSpec(
            name="w",
            shape="scalar",
            output_mode="scalar",
            source_name="w",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
)

__all__ = ["PDE_SPEC"]

"""Schnakenberg PDE spec."""

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

PDE_SPEC = PDESpec(
    name="schnakenberg",
    category="physics",
    description=(
        "Schnakenberg two-species reaction-diffusion system for Turing patterns."
    ),
    equations={
        "u": "du/dt = Du * laplacian(u) + a - u + u^2 * v",
        "v": "dv/dt = Dv * laplacian(v) + b - u^2 * v",
    },
    parameters=[
        PDEParameter(
            "Du",
            "Diffusion coefficient of activator u",
            hard_min=0.0,
            sampling_min=0.004,
            sampling_max=0.012,
        ),
        PDEParameter(
            "Dv",
            "Diffusion coefficient of inhibitor v",
            hard_min=0.0,
            sampling_min=0.14,
            sampling_max=0.3,
        ),
        PDEParameter(
            "a",
            "Activator production rate",
            hard_min=0.0,
            sampling_min=0.08,
            sampling_max=0.16,
        ),
        PDEParameter(
            "b",
            "Inhibitor production rate",
            hard_min=0.0,
            sampling_min=0.75,
            sampling_max=1.05,
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
    },
    boundary_fields={
        "u": BoundaryFieldSpec(
            name="u",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for activator u.",
        ),
        "v": BoundaryFieldSpec(
            name="v",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for inhibitor v.",
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
    },
    static_fields=[],
    supported_dimensions=[2],
)

__all__ = ["PDE_SPEC"]

"""Fitzhugh nagumo PDE spec."""

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    GENERIC_STOCHASTIC_COUPLINGS,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

PDE_SPEC = PDESpec(
    name="fitzhugh_nagumo",
    category="biology",
    description="FitzHugh-Nagumo reaction-diffusion system for excitable dynamics.",
    equations={
        "u": "du/dt = Du * laplacian(u) + u - u^3 - v",
        "v": "tau * dv/dt = Dv * laplacian(v) + u - b * v + a",
    },
    parameters=[
        PDEParameter(
            "Du",
            "Diffusion coefficient of activator u",
            hard_min=0.0,
            sampling_min=0.008,
            sampling_max=0.018,
        ),
        PDEParameter(
            "Dv",
            "Diffusion coefficient of inhibitor v",
            hard_min=0.0,
            sampling_min=0.03,
            sampling_max=0.08,
        ),
        PDEParameter(
            "tau",
            "Inhibitor timescale ratio",
            hard_min=0.0,
            sampling_min=0.8,
            sampling_max=1.3,
        ),
        PDEParameter(
            "b",
            "Recovery sensitivity to activator",
            hard_min=0.0,
            sampling_min=0.7,
            sampling_max=1.0,
        ),
        PDEParameter(
            "a",
            "Threshold / offset parameter",
            hard_min=0.0,
            sampling_min=0.05,
            sampling_max=0.18,
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
            stochastic_couplings=GENERIC_STOCHASTIC_COUPLINGS,
        ),
        "v": StateSpec(
            name="v",
            shape="scalar",
            stochastic_couplings=GENERIC_STOCHASTIC_COUPLINGS,
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

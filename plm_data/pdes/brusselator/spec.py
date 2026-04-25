"""Brusselator PDE spec."""

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
    name="brusselator",
    category="physics",
    description=("Brusselator reaction-diffusion system for Turing pattern formation."),
    equations={
        "u": "du/dt = Du * laplacian(u) + a - (b + 1) * u + u^2 * v",
        "v": "dv/dt = Dv * laplacian(v) + b * u - u^2 * v",
    },
    parameters=[
        PDEParameter(
            "Du",
            "Diffusion coefficient of species u",
            hard_min=0.0,
            sampling_min=0.006,
            sampling_max=0.014,
        ),
        PDEParameter(
            "Dv",
            "Diffusion coefficient of species v",
            hard_min=0.0,
            sampling_min=0.06,
            sampling_max=0.14,
        ),
        PDEParameter(
            "a",
            "Feed kinetic parameter",
            hard_min=0.0,
            sampling_min=0.9,
            sampling_max=1.15,
        ),
        PDEParameter(
            "b",
            "Reaction kinetic parameter",
            hard_min=0.0,
            sampling_min=2.2,
            sampling_max=3.0,
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
            description="Boundary conditions for species u.",
        ),
        "v": BoundaryFieldSpec(
            name="v",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for species v.",
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

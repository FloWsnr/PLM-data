"""Lorenz PDE spec."""

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
    name="lorenz",
    category="physics",
    description="Diffusively coupled Lorenz oscillator reaction-diffusion system.",
    equations={
        "x": "dx/dt = D * laplacian(x) + sigma * (y - x)",
        "y": "dy/dt = D * laplacian(y) + x * (rho - z) - y",
        "z": "dz/dt = D * laplacian(z) + x * y - beta * z",
    },
    parameters=[
        PDEParameter(
            "sigma",
            "Prandtl number",
            hard_min=0.0,
            sampling_min=7.5,
            sampling_max=11.0,
        ),
        PDEParameter(
            "rho",
            "Normalized Rayleigh number",
            hard_min=0.0,
            sampling_min=27.0,
            sampling_max=36.0,
        ),
        PDEParameter(
            "beta",
            "Geometric factor",
            hard_min=0.0,
            sampling_min=2.45,
            sampling_max=2.9,
        ),
        PDEParameter(
            "D",
            "Diffusive spatial coupling coefficient",
            hard_min=0.0,
            sampling_min=0.08,
            sampling_max=0.22,
        ),
    ],
    inputs={
        "x": InputSpec(
            name="x",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "y": InputSpec(
            name="y",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "z": InputSpec(
            name="z",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "x": BoundaryFieldSpec(
            name="x",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for Lorenz x field.",
        ),
        "y": BoundaryFieldSpec(
            name="y",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for Lorenz y field.",
        ),
        "z": BoundaryFieldSpec(
            name="z",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for Lorenz z field.",
        ),
    },
    states={
        "x": StateSpec(
            name="x",
            shape="scalar",
            stochastic_couplings=GENERIC_STOCHASTIC_COUPLINGS,
        ),
        "y": StateSpec(
            name="y",
            shape="scalar",
            stochastic_couplings=GENERIC_STOCHASTIC_COUPLINGS,
        ),
        "z": StateSpec(
            name="z",
            shape="scalar",
            stochastic_couplings=GENERIC_STOCHASTIC_COUPLINGS,
        ),
    },
    outputs={
        "x": OutputSpec(
            name="x",
            shape="scalar",
            output_mode="scalar",
            source_name="x",
        ),
        "y": OutputSpec(
            name="y",
            shape="scalar",
            output_mode="scalar",
            source_name="y",
        ),
        "z": OutputSpec(
            name="z",
            shape="scalar",
            output_mode="scalar",
            source_name="z",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
)

__all__ = ["PDE_SPEC"]

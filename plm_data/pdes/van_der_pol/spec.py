"""Van der pol PDE spec."""

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
    name="van_der_pol",
    category="physics",
    description="Diffusively coupled Van der Pol oscillator reaction-diffusion system.",
    equations={
        "u": "du/dt = Du * laplacian(u) + v",
        "v": "dv/dt = Dv * laplacian(v) + mu * (1 - u^2) * v - u",
    },
    parameters=[
        PDEParameter(
            "Du",
            "Diffusion coefficient of displacement u",
            hard_min=0.0,
            sampling_min=0.08,
            sampling_max=0.2,
        ),
        PDEParameter(
            "Dv",
            "Diffusion coefficient of velocity v",
            hard_min=0.0,
            sampling_min=0.015,
            sampling_max=0.06,
        ),
        PDEParameter(
            "mu",
            "Nonlinear damping coefficient",
            hard_min=0.0,
            sampling_min=0.8,
            sampling_max=2.2,
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
            description="Boundary conditions for displacement u.",
        ),
        "v": BoundaryFieldSpec(
            name="v",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for velocity v.",
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

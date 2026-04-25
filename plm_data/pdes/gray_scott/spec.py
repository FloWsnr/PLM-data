"""Gray scott PDE spec."""

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    CoefficientSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    SATURATING_STOCHASTIC_COUPLINGS,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

PDE_SPEC = PDESpec(
    name="gray_scott",
    category="physics",
    description="Gray-Scott reaction-diffusion system for substrate/autocatalyst patterns.",
    equations={
        "u": "du/dt + velocity dot grad(u) = Du * laplacian(u) - u * v^2 + F * (1 - u)",
        "v": "dv/dt = Dv * laplacian(v) + u * v^2 - (F + k) * v",
    },
    parameters=[
        PDEParameter(
            "Du",
            "Diffusion coefficient of substrate u",
            hard_min=0.0,
            sampling_min=0.08,
            sampling_max=0.16,
        ),
        PDEParameter(
            "Dv",
            "Diffusion coefficient of autocatalyst v",
            hard_min=0.0,
            sampling_min=0.04,
            sampling_max=0.08,
        ),
        PDEParameter(
            "F",
            "Feed rate",
            hard_min=0.0,
            sampling_min=0.026,
            sampling_max=0.045,
        ),
        PDEParameter(
            "k",
            "Kill rate",
            hard_min=0.0,
            sampling_min=0.052,
            sampling_max=0.068,
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
            description="Boundary conditions for substrate u.",
        ),
        "v": BoundaryFieldSpec(
            name="v",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for autocatalyst v.",
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
    coefficients={
        "velocity": CoefficientSpec(
            name="velocity",
            shape="vector",
            description="Prescribed advection velocity applied to substrate u.",
        )
    },
)

__all__ = ["PDE_SPEC"]

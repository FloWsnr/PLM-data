"""Klausmeier topography PDE spec."""

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
    name="klausmeier_topography",
    category="biology",
    description=(
        "Klausmeier vegetation model coupling water and biomass on static terrain."
    ),
    equations={
        "w": "dw/dt = a - w - w * n^2 + D * laplacian(w) + V * div(w * grad(T))",
        "n": "dn/dt = w * n^2 - m * n + Dn * laplacian(n)",
    },
    parameters=[
        PDEParameter(
            "a",
            "Rainfall rate",
            hard_min=0.0,
            sampling_min=0.85,
            sampling_max=1.2,
        ),
        PDEParameter(
            "m",
            "Plant mortality rate",
            hard_min=0.0,
            sampling_min=0.35,
            sampling_max=0.55,
        ),
        PDEParameter(
            "D",
            "Water diffusion coefficient",
            hard_min=0.0,
            sampling_min=0.04,
            sampling_max=0.1,
        ),
        PDEParameter(
            "Dn",
            "Vegetation diffusion coefficient",
            hard_min=0.0,
            sampling_min=0.002,
            sampling_max=0.008,
        ),
        PDEParameter(
            "V",
            "Topography advection strength",
            hard_min=0.0,
            sampling_min=0.08,
            sampling_max=0.18,
        ),
    ],
    inputs={
        "w": InputSpec(
            name="w",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "n": InputSpec(
            name="n",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "w": BoundaryFieldSpec(
            name="w",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for water w.",
        ),
        "n": BoundaryFieldSpec(
            name="n",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for vegetation n.",
        ),
    },
    states={
        "w": StateSpec(
            name="w",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
        "n": StateSpec(
            name="n",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
    },
    outputs={
        "w": OutputSpec(
            name="w",
            shape="scalar",
            output_mode="scalar",
            source_name="w",
        ),
        "n": OutputSpec(
            name="n",
            shape="scalar",
            output_mode="scalar",
            source_name="n",
        ),
        "topography": OutputSpec(
            name="topography",
            shape="scalar",
            output_mode="scalar",
            source_name="topography",
            source_kind="derived",
        ),
    },
    static_fields=["topography"],
    supported_dimensions=[2],
    coefficients={
        "topography": CoefficientSpec(
            name="topography",
            shape="scalar",
            description="Static terrain height T(x, y).",
            allow_randomization=True,
        )
    },
)

__all__ = ["PDE_SPEC"]

"""Keller segel PDE spec."""

from plm_data.pdes.metadata import SCALAR_STANDARD_BOUNDARY_OPERATORS

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    SATURATING_STOCHASTIC_COUPLINGS,
    StateSpec,
)

_KELLER_SEGEL_RHO_BOUNDARY_OPERATORS = SCALAR_STANDARD_BOUNDARY_OPERATORS
_KELLER_SEGEL_C_BOUNDARY_OPERATORS = {
    name: SCALAR_STANDARD_BOUNDARY_OPERATORS[name]
    for name in ("neumann", "robin", "periodic")
}

PDE_SPEC = PDESpec(
    name="keller_segel",
    category="biology",
    description=(
        "Keller-Segel chemotaxis model with logistic growth and receptor saturation."
    ),
    equations={
        "rho": (
            "drho/dt = D_rho * laplacian(rho) "
            "- div(chi0 * rho / (1 + rho^2) * grad(c)) + r * rho * (1 - rho)"
        ),
        "c": "dc/dt = D_c * laplacian(c) + alpha * rho - beta * c",
    },
    parameters=[
        PDEParameter(
            "D_rho",
            "Cell diffusivity",
            hard_min=0.0,
            sampling_min=0.004,
            sampling_max=0.012,
        ),
        PDEParameter(
            "D_c",
            "Chemoattractant diffusivity",
            hard_min=0.0,
            sampling_min=0.035,
            sampling_max=0.09,
        ),
        PDEParameter(
            "chi0",
            "Chemotactic coefficient",
            hard_min=0.0,
            sampling_min=0.06,
            sampling_max=0.16,
        ),
        PDEParameter(
            "alpha",
            "Chemoattractant production rate",
            hard_min=0.0,
            sampling_min=0.18,
            sampling_max=0.38,
        ),
        PDEParameter(
            "beta",
            "Chemoattractant degradation rate",
            hard_min=0.0,
            sampling_min=0.18,
            sampling_max=0.42,
        ),
        PDEParameter(
            "r",
            "Logistic growth rate",
            hard_min=0.0,
            sampling_min=0.03,
            sampling_max=0.12,
        ),
    ],
    inputs={
        "rho": InputSpec(
            name="rho",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "c": InputSpec(
            name="c",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "rho": BoundaryFieldSpec(
            name="rho",
            shape="scalar",
            operators=_KELLER_SEGEL_RHO_BOUNDARY_OPERATORS,
            description="Boundary conditions for cell density rho.",
        ),
        "c": BoundaryFieldSpec(
            name="c",
            shape="scalar",
            operators=_KELLER_SEGEL_C_BOUNDARY_OPERATORS,
            description="Boundary conditions for chemoattractant c.",
        ),
    },
    states={
        "rho": StateSpec(
            name="rho",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
        "c": StateSpec(
            name="c",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
    },
    outputs={
        "rho": OutputSpec(
            name="rho",
            shape="scalar",
            output_mode="scalar",
            source_name="rho",
        ),
        "c": OutputSpec(
            name="c",
            shape="scalar",
            output_mode="scalar",
            source_name="c",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
)

__all__ = ["PDE_SPEC"]

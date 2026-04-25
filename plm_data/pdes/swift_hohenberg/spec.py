"""Swift hohenberg PDE spec."""

from plm_data.boundary_conditions import get_boundary_operator_spec
from plm_data.pdes.metadata import SCALAR_STANDARD_BOUNDARY_OPERATORS

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    CoefficientSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    StateSpec,
)

_SWIFT_HOHENBERG_BOUNDARY_OPERATORS = {
    "periodic": SCALAR_STANDARD_BOUNDARY_OPERATORS["periodic"],
    "simply_supported": get_boundary_operator_spec("simply_supported"),
}

PDE_SPEC = PDESpec(
    name="swift_hohenberg",
    category="physics",
    description=(
        "Swift-Hohenberg equation for pattern formation using a mixed u/w "
        "formulation where w = (q0^2 + laplacian)(u)."
    ),
    equations={
        "u": (
            "du/dt + velocity.grad(u) = r*u - q0^2*w - laplacian(w) "
            "+ alpha*u^2 + beta*u^3 + gamma*u^5"
        ),
        "w": "w = q0^2*u + laplacian(u)",
    },
    parameters=[
        PDEParameter(
            "r",
            "Bifurcation parameter",
            sampling_min=0.08,
            sampling_max=0.22,
        ),
        PDEParameter(
            "q0",
            "Critical wavenumber",
            hard_min=0.0,
            sampling_min=0.85,
            sampling_max=1.15,
        ),
        PDEParameter(
            "alpha",
            "Quadratic nonlinearity coefficient",
            sampling_min=-0.05,
            sampling_max=0.05,
        ),
        PDEParameter(
            "beta",
            "Cubic nonlinearity coefficient",
            sampling_min=-1.0,
            sampling_max=-0.55,
        ),
        PDEParameter(
            "gamma",
            "Quintic nonlinearity coefficient",
            sampling_min=-0.04,
            sampling_max=0.0,
        ),
        PDEParameter(
            "theta",
            "Time-stepping parameter",
            hard_min=0.0,
            hard_max=1.0,
            default=0.5,
        ),
    ],
    inputs={
        "u": InputSpec(
            name="u",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        )
    },
    boundary_fields={
        "u": BoundaryFieldSpec(
            name="u",
            shape="scalar",
            operators=_SWIFT_HOHENBERG_BOUNDARY_OPERATORS,
            description="Boundary conditions for the primary field u.",
        )
    },
    states={
        "u": StateSpec(name="u", shape="scalar"),
        "w": StateSpec(name="w", shape="scalar"),
    },
    outputs={
        "u": OutputSpec(
            name="u",
            shape="scalar",
            output_mode="scalar",
            source_name="u",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
    coefficients={
        "velocity": CoefficientSpec(
            name="velocity",
            shape="vector",
            description="Prescribed advection velocity for the primary field u.",
        )
    },
)

__all__ = ["PDE_SPEC"]

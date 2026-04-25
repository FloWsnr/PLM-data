"""Zakharov kuznetsov PDE spec."""

from plm_data.pdes.metadata import SCALAR_STANDARD_BOUNDARY_OPERATORS

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    StateSpec,
)

_ZK_BOUNDARY_OPERATORS = {
    name: SCALAR_STANDARD_BOUNDARY_OPERATORS[name]
    for name in ("dirichlet", "neumann", "periodic")
}

PDE_SPEC = PDESpec(
    name="zakharov_kuznetsov",
    category="physics",
    description=(
        "Zakharov-Kuznetsov equation for dispersive nonlinear waves using a "
        "mixed u/w formulation where w = -laplacian(u)."
    ),
    equations={
        "u": "du/dt + alpha*u*du/dx - dw/dx = 0",
        "w": "w + laplacian(u) = 0",
    },
    parameters=[
        PDEParameter(
            "alpha",
            "Nonlinear advection coefficient",
            sampling_min=1.5,
            sampling_max=3.5,
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
            operators=_ZK_BOUNDARY_OPERATORS,
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
)

__all__ = ["PDE_SPEC"]

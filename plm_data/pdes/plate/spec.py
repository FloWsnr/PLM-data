"""Plate PDE spec."""

from plm_data.boundary_conditions import get_boundary_operator_spec

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    CoefficientSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    StateSpec,
)

_PLATE_BOUNDARY_OPERATORS = {
    "simply_supported": get_boundary_operator_spec("simply_supported"),
}

PDE_SPEC = PDESpec(
    name="plate",
    category="basic",
    description=(
        "Kirchhoff plate equation using a mixed deflection/velocity/moment "
        "formulation with simply supported boundaries."
    ),
    equations={
        "deflection": "dw/dt = v",
        "velocity": "rho_h * dv/dt + damping * v - div(rigidity grad(m)) = q",
        "moment": "m = -laplacian(w)",
    },
    parameters=[
        PDEParameter(
            "theta",
            "Implicit time-stepping parameter in [0.5, 1.0]",
            hard_min=0.5,
            hard_max=1.0,
            sampling_min=0.55,
            sampling_max=0.75,
        ),
    ],
    inputs={
        "deflection": InputSpec(
            name="deflection",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "velocity": InputSpec(
            name="velocity",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "load": InputSpec(
            name="load",
            shape="scalar",
            allow_source=True,
            allow_initial_condition=False,
        ),
    },
    boundary_fields={
        "deflection": BoundaryFieldSpec(
            name="deflection",
            shape="scalar",
            operators=_PLATE_BOUNDARY_OPERATORS,
            description="Plate edge conditions for the deflection field.",
        )
    },
    states={
        "deflection": StateSpec(name="deflection", shape="scalar"),
        "velocity": StateSpec(name="velocity", shape="scalar"),
        "moment": StateSpec(name="moment", shape="scalar"),
    },
    outputs={
        "deflection": OutputSpec(
            name="deflection",
            shape="scalar",
            output_mode="scalar",
            source_name="deflection",
        ),
        "velocity": OutputSpec(
            name="velocity",
            shape="scalar",
            output_mode="scalar",
            source_name="velocity",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
    coefficients={
        "rho_h": CoefficientSpec(
            name="rho_h",
            shape="scalar",
            description="Mass per unit area.",
        ),
        "damping": CoefficientSpec(
            name="damping",
            shape="scalar",
            description="Viscous damping coefficient.",
        ),
        "rigidity": CoefficientSpec(
            name="rigidity",
            shape="scalar",
            description="Flexural rigidity coefficient.",
        ),
    },
)

__all__ = ["PDE_SPEC"]

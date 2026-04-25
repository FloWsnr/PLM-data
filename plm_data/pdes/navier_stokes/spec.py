"""Navier stokes PDE spec."""

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    StateSpec,
    VECTOR_STANDARD_BOUNDARY_OPERATORS,
)

PDE_SPEC = PDESpec(
    name="navier_stokes",
    category="fluids",
    description="Incompressible Navier-Stokes equations.",
    equations={
        "velocity": "du/dt + (u.grad)u = -grad(p) + (1/Re)*laplacian(u)",
        "pressure": "div(u) = 0",
    },
    parameters=[
        PDEParameter(
            "Re",
            "Reynolds number",
            hard_min=0.0,
            sampling_min=18.0,
            sampling_max=45.0,
        ),
        PDEParameter(
            "k",
            "Polynomial degree parameter",
            kind="int",
            hard_min=1,
            default=1,
        ),
    ],
    inputs={
        "velocity": InputSpec(
            name="velocity",
            shape="vector",
            allow_source=True,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "velocity": BoundaryFieldSpec(
            name="velocity",
            shape="vector",
            operators=VECTOR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the velocity field.",
        )
    },
    states={
        "velocity": StateSpec(name="velocity", shape="vector"),
        "pressure": StateSpec(name="pressure", shape="scalar"),
    },
    outputs={
        "velocity": OutputSpec(
            name="velocity",
            shape="vector",
            output_mode="components",
            source_name="velocity",
        ),
        "pressure": OutputSpec(
            name="pressure",
            shape="scalar",
            output_mode="scalar",
            source_name="pressure",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
)

__all__ = ["PDE_SPEC"]

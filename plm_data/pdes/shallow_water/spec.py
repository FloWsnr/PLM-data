"""Shallow water PDE spec."""

from plm_data.pdes.metadata import SCALAR_STANDARD_BOUNDARY_OPERATORS
from plm_data.pdes.metadata import VECTOR_STANDARD_BOUNDARY_OPERATORS

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    CoefficientSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    StateSpec,
)

_SHALLOW_WATER_HEIGHT_BOUNDARY_OPERATORS = {
    "dirichlet": SCALAR_STANDARD_BOUNDARY_OPERATORS["dirichlet"],
    "neumann": SCALAR_STANDARD_BOUNDARY_OPERATORS["neumann"],
    "periodic": SCALAR_STANDARD_BOUNDARY_OPERATORS["periodic"],
}
_SHALLOW_WATER_VELOCITY_BOUNDARY_OPERATORS = {
    "dirichlet": VECTOR_STANDARD_BOUNDARY_OPERATORS["dirichlet"],
    "neumann": VECTOR_STANDARD_BOUNDARY_OPERATORS["neumann"],
    "periodic": VECTOR_STANDARD_BOUNDARY_OPERATORS["periodic"],
}

PDE_SPEC = PDESpec(
    name="shallow_water",
    category="fluids",
    description="Two-dimensional shallow-water equations for height and velocity.",
    equations={
        "height": "d(height)/dt + div((mean_depth + height) * velocity) = 0",
        "velocity": (
            "d(velocity)/dt + (velocity.grad)velocity + gravity*grad(height + "
            "bathymetry) = -drag*velocity + viscosity*laplacian(velocity) "
            "- coriolis*perp(velocity)"
        ),
    },
    parameters=[
        PDEParameter(
            "gravity",
            "Gravitational acceleration coefficient",
            hard_min=0.0,
            sampling_min=0.7,
            sampling_max=1.2,
        ),
        PDEParameter(
            "mean_depth",
            "Background water depth",
            hard_min=0.0,
            sampling_min=0.8,
            sampling_max=1.2,
        ),
        PDEParameter(
            "drag",
            "Linear bottom-friction coefficient",
            hard_min=0.0,
            sampling_min=0.015,
            sampling_max=0.05,
        ),
        PDEParameter(
            "viscosity",
            "Velocity diffusion coefficient",
            hard_min=0.0,
            sampling_min=0.002,
            sampling_max=0.006,
        ),
        PDEParameter(
            "coriolis",
            "Coriolis rotation coefficient",
            sampling_min=-0.08,
            sampling_max=0.08,
        ),
    ],
    inputs={
        "height": InputSpec(
            name="height",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "velocity": InputSpec(
            name="velocity",
            shape="vector",
            allow_source=False,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "height": BoundaryFieldSpec(
            name="height",
            shape="scalar",
            operators=_SHALLOW_WATER_HEIGHT_BOUNDARY_OPERATORS,
            description="Boundary conditions for the height anomaly.",
        ),
        "velocity": BoundaryFieldSpec(
            name="velocity",
            shape="vector",
            operators=_SHALLOW_WATER_VELOCITY_BOUNDARY_OPERATORS,
            description="Boundary conditions for depth-averaged velocity.",
        ),
    },
    states={
        "height": StateSpec(name="height", shape="scalar"),
        "velocity": StateSpec(name="velocity", shape="vector"),
    },
    outputs={
        "height": OutputSpec(
            name="height",
            shape="scalar",
            output_mode="scalar",
            source_name="height",
        ),
        "velocity": OutputSpec(
            name="velocity",
            shape="vector",
            output_mode="components",
            source_name="velocity",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
    coefficients={
        "bathymetry": CoefficientSpec(
            name="bathymetry",
            shape="scalar",
            description="Bed-elevation field added to the hydrostatic pressure term.",
        )
    },
)

__all__ = ["PDE_SPEC"]

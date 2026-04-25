"""Thermal convection PDE spec."""

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
    VECTOR_STANDARD_BOUNDARY_OPERATORS,
)

PDE_SPEC = PDESpec(
    name="thermal_convection",
    category="fluids",
    description="Thermal convection using the Boussinesq Rayleigh-Benard equations.",
    equations={
        "velocity": (
            "du/dt + (u_prev.grad)u = -grad(p) + sqrt(Pr/Ra)*laplacian(u) + T*e_y"
        ),
        "pressure": "div(u) = 0",
        "temperature": "dT/dt + u_prev.grad(T) = (1/sqrt(Ra*Pr))*laplacian(T) + f_T",
    },
    parameters=[
        PDEParameter(
            "Ra",
            "Rayleigh number",
            hard_min=0.0,
            sampling_min=120.0,
            sampling_max=320.0,
        ),
        PDEParameter(
            "Pr",
            "Prandtl number",
            hard_min=0.0,
            sampling_min=0.8,
            sampling_max=1.4,
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
        "temperature": InputSpec(
            name="temperature",
            shape="scalar",
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
        ),
        "temperature": BoundaryFieldSpec(
            name="temperature",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the temperature field.",
        ),
    },
    states={
        "velocity": StateSpec(name="velocity", shape="vector"),
        "pressure": StateSpec(name="pressure", shape="scalar"),
        "temperature": StateSpec(name="temperature", shape="scalar"),
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
        "temperature": OutputSpec(
            name="temperature",
            shape="scalar",
            output_mode="scalar",
            source_name="temperature",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
)

__all__ = ["PDE_SPEC"]

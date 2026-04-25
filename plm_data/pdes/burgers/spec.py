"""Burgers PDE spec."""

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    GENERIC_STOCHASTIC_COUPLINGS,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    StateSpec,
    VECTOR_STANDARD_BOUNDARY_OPERATORS,
)

PDE_SPEC = PDESpec(
    name="burgers",
    category="fluids",
    description=(
        "Vector viscous Burgers equation with nonlinear self-advection and diffusion."
    ),
    equations={
        "velocity": "du/dt + (u dot grad)u = nu * laplacian(u) + f",
    },
    parameters=[
        PDEParameter(
            "nu",
            "Kinematic viscosity / diffusion coefficient",
            hard_min=0.0,
            sampling_min=0.02,
            sampling_max=0.05,
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
            description="Boundary conditions for the Burgers velocity field.",
        ),
    },
    states={
        "velocity": StateSpec(
            name="velocity",
            shape="vector",
            stochastic_couplings=GENERIC_STOCHASTIC_COUPLINGS,
        ),
    },
    outputs={
        "velocity": OutputSpec(
            name="velocity",
            shape="vector",
            output_mode="components",
            source_name="velocity",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
)

__all__ = ["PDE_SPEC"]

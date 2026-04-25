"""Advection PDE spec."""

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    CoefficientSpec,
    GENERIC_STOCHASTIC_COUPLINGS,
    InputSpec,
    OutputSpec,
    PDESpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

PDE_SPEC = PDESpec(
    name="advection",
    category="basic",
    description=(
        "Scalar advection-diffusion equation with a prescribed velocity field "
        "and SUPG stabilization."
    ),
    equations={
        "u": "du/dt + velocity dot grad(u) = div(diffusivity grad(u)) + f",
    },
    parameters=[],
    inputs={
        "u": InputSpec(
            name="u",
            shape="scalar",
            allow_source=True,
            allow_initial_condition=True,
        )
    },
    boundary_fields={
        "u": BoundaryFieldSpec(
            name="u",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the transported scalar field.",
        )
    },
    states={
        "u": StateSpec(
            name="u",
            shape="scalar",
            stochastic_couplings=GENERIC_STOCHASTIC_COUPLINGS,
        )
    },
    outputs={
        "u": OutputSpec(
            name="u",
            shape="scalar",
            output_mode="scalar",
            source_name="u",
        )
    },
    static_fields=[],
    supported_dimensions=[2],
    coefficients={
        "velocity": CoefficientSpec(
            name="velocity",
            shape="vector",
            description="Prescribed advection velocity field.",
        ),
        "diffusivity": CoefficientSpec(
            name="diffusivity",
            shape="scalar",
            description="Scalar diffusion coefficient field.",
            allow_randomization=True,
        ),
    },
)

__all__ = ["PDE_SPEC"]

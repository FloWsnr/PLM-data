"""Heat PDE spec."""

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
    name="heat",
    category="basic",
    description="Heat equation du/dt = div(kappa * grad(u)) + f",
    equations={"u": "∂u/∂t = ∇·(κ ∇u) + f"},
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
            description="Boundary conditions for the scalar temperature field.",
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
        "kappa": CoefficientSpec(
            name="kappa",
            shape="scalar",
            description="Thermal diffusivity coefficient field.",
            allow_randomization=True,
        )
    },
)

__all__ = ["PDE_SPEC"]

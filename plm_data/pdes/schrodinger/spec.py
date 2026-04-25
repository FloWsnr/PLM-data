"""Schrodinger PDE spec."""

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

_SCHRODINGER_BOUNDARY_OPERATORS = {
    name: SCALAR_STANDARD_BOUNDARY_OPERATORS[name] for name in ("dirichlet", "periodic")
}

PDE_SPEC = PDESpec(
    name="schrodinger",
    category="basic",
    description=(
        "Linear time-dependent Schrodinger equation solved via a real/imaginary "
        "split with a configurable external potential."
    ),
    equations={
        "u": "du/dt = D * lap(v) + potential * v",
        "v": "dv/dt = -D * lap(u) - potential * u",
    },
    parameters=[
        PDEParameter(
            "D",
            "Dispersion coefficient",
            hard_min=0.0,
            sampling_min=0.025,
            sampling_max=0.06,
        ),
        PDEParameter(
            "theta",
            "Time-stepping parameter in [0.5, 1.0]",
            hard_min=0.5,
            hard_max=1.0,
            sampling_min=0.5,
            sampling_max=0.6,
        ),
    ],
    inputs={
        "u": InputSpec(
            name="u",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "v": InputSpec(
            name="v",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "u": BoundaryFieldSpec(
            name="u",
            shape="scalar",
            operators=_SCHRODINGER_BOUNDARY_OPERATORS,
            description="Boundary conditions for the real component.",
        ),
        "v": BoundaryFieldSpec(
            name="v",
            shape="scalar",
            operators=_SCHRODINGER_BOUNDARY_OPERATORS,
            description="Boundary conditions for the imaginary component.",
        ),
    },
    states={
        "u": StateSpec(name="u", shape="scalar"),
        "v": StateSpec(name="v", shape="scalar"),
    },
    outputs={
        "u": OutputSpec(
            name="u",
            shape="scalar",
            output_mode="scalar",
            source_name="u",
        ),
        "v": OutputSpec(
            name="v",
            shape="scalar",
            output_mode="scalar",
            source_name="v",
        ),
        "density": OutputSpec(
            name="density",
            shape="scalar",
            output_mode="scalar",
            source_name="density",
            source_kind="derived",
        ),
        "potential": OutputSpec(
            name="potential",
            shape="scalar",
            output_mode="scalar",
            source_name="potential",
            source_kind="derived",
        ),
    },
    static_fields=["potential"],
    supported_dimensions=[2],
    coefficients={
        "potential": CoefficientSpec(
            name="potential",
            shape="scalar",
            description="External potential field.",
            allow_randomization=True,
        )
    },
)

__all__ = ["PDE_SPEC"]

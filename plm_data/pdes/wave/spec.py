"""Wave PDE spec."""

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    CoefficientSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

PDE_SPEC = PDESpec(
    name="wave",
    category="basic",
    description="Damped wave equation with scalar displacement and velocity outputs.",
    equations={
        "u": "d2u/dt2 + damping * du/dt = div(c_sq grad(u)) + f",
        "v": "v = du/dt",
    },
    parameters=[
        PDEParameter(
            "damping",
            "Linear damping coefficient",
            hard_min=0.0,
            sampling_min=0.04,
            sampling_max=0.16,
        )
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
        "forcing": InputSpec(
            name="forcing",
            shape="scalar",
            allow_source=True,
            allow_initial_condition=False,
        ),
    },
    boundary_fields={
        "u": BoundaryFieldSpec(
            name="u",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the scalar displacement field.",
        )
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
    },
    static_fields=[],
    supported_dimensions=[2],
    coefficients={
        "c_sq": CoefficientSpec(
            name="c_sq",
            shape="scalar",
            description="Wave-speed-squared coefficient field.",
            allow_randomization=True,
        )
    },
)

__all__ = ["PDE_SPEC"]

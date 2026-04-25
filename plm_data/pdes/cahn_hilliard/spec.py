"""Cahn hilliard PDE spec."""

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

PDE_SPEC = PDESpec(
    name="cahn_hilliard",
    category="physics",
    description=(
        "Cahn-Hilliard equation for phase separation using a mixed "
        "concentration/chemical-potential formulation."
    ),
    equations={
        "c": "dc/dt = div(M * grad(mu))",
        "mu": "mu = df/dc - lambda * laplacian(c)",
    },
    parameters=[
        PDEParameter(
            "lmbda",
            "Surface parameter (interface width)",
            hard_min=0.0,
            sampling_min=0.008,
            sampling_max=0.02,
        ),
        PDEParameter(
            "barrier_height",
            "Height of the double-well free-energy barrier",
            hard_min=0.0,
            sampling_min=0.8,
            sampling_max=1.6,
        ),
        PDEParameter(
            "mobility",
            "Mobility coefficient M",
            hard_min=0.0,
            sampling_min=0.006,
            sampling_max=0.018,
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
        "c": InputSpec(
            name="c",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        )
    },
    boundary_fields={
        "c": BoundaryFieldSpec(
            name="c",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description=(
                "Boundary conditions for concentration. Periodic side pairs are "
                "constrained strongly; homogeneous Neumann entries mark the natural "
                "no-flux boundary condition."
            ),
        )
    },
    states={
        "c": StateSpec(name="c", shape="scalar"),
        "mu": StateSpec(name="mu", shape="scalar"),
    },
    outputs={
        "c": OutputSpec(
            name="c",
            shape="scalar",
            output_mode="scalar",
            source_name="c",
        )
    },
    static_fields=[],
    supported_dimensions=[2],
)

__all__ = ["PDE_SPEC"]

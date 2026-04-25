"""Cgl PDE spec."""

from plm_data.pdes.metadata import SCALAR_STANDARD_BOUNDARY_OPERATORS

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    StateSpec,
)

_CGL_BOUNDARY_OPERATORS = {
    name: SCALAR_STANDARD_BOUNDARY_OPERATORS[name]
    for name in ("dirichlet", "neumann", "periodic")
}

PDE_SPEC = PDESpec(
    name="cgl",
    category="physics",
    description=(
        "Complex Ginzburg-Landau equation solved via real/imaginary split, "
        "covering canonical CGL and NLS-like dynamics as parameter cases."
    ),
    equations={
        "u": (
            "du/dt = D_r*lap(u) - D_i*lap(v) + a_r*u - a_i*v + |A|^2*(b_r*u - b_i*v)"
        ),
        "v": (
            "dv/dt = D_i*lap(u) + D_r*lap(v) + a_i*u + a_r*v + |A|^2*(b_i*u + b_r*v)"
        ),
    },
    parameters=[
        PDEParameter(
            "D_r",
            "Real part of diffusion coefficient",
            hard_min=0.0,
            sampling_min=0.05,
            sampling_max=0.12,
        ),
        PDEParameter(
            "D_i",
            "Imaginary part of diffusion coefficient",
            sampling_min=0.12,
            sampling_max=0.35,
        ),
        PDEParameter(
            "a_r",
            "Real part of linear coefficient",
            sampling_min=0.12,
            sampling_max=0.35,
        ),
        PDEParameter(
            "a_i",
            "Imaginary part of linear coefficient",
            sampling_min=-0.08,
            sampling_max=0.08,
        ),
        PDEParameter(
            "b_r",
            "Real part of nonlinear coefficient",
            sampling_min=-0.8,
            sampling_max=-0.35,
        ),
        PDEParameter(
            "b_i",
            "Imaginary part of nonlinear coefficient",
            sampling_min=-0.45,
            sampling_max=0.1,
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
            operators=_CGL_BOUNDARY_OPERATORS,
            description="Boundary conditions for Re(A).",
        ),
        "v": BoundaryFieldSpec(
            name="v",
            shape="scalar",
            operators=_CGL_BOUNDARY_OPERATORS,
            description="Boundary conditions for Im(A).",
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
        "amplitude": OutputSpec(
            name="amplitude",
            shape="scalar",
            output_mode="scalar",
            source_name="amplitude",
            source_kind="derived",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
)

__all__ = ["PDE_SPEC"]

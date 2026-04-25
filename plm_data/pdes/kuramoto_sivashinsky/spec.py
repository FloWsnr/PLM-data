"""Kuramoto sivashinsky PDE spec."""

from plm_data.pdes.metadata import SCALAR_STANDARD_BOUNDARY_OPERATORS

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    StateSpec,
)

_KS_BOUNDARY_OPERATORS = {
    name: SCALAR_STANDARD_BOUNDARY_OPERATORS[name]
    for name in ("dirichlet", "neumann", "periodic")
}

PDE_SPEC = PDESpec(
    name="kuramoto_sivashinsky",
    category="physics",
    description=(
        "Kuramoto-Sivashinsky equation for spatiotemporal chaos using a mixed "
        "u/v formulation where v = -laplacian(u)."
    ),
    equations={
        "u": (
            "du/dt + velocity.grad(u) = hyperdiffusion*laplacian(v) "
            "+ anti_diffusion*v - (nonlinear_strength/2)|grad(u)|^2 - damping*u"
        ),
        "v": "v + laplacian(u) = 0",
    },
    parameters=[
        PDEParameter(
            "theta",
            "Time-stepping parameter",
            hard_min=0.0,
            hard_max=1.0,
            default=0.5,
        ),
        PDEParameter(
            "hyperdiffusion",
            "Stabilizing fourth-order coefficient",
            hard_min=0.0,
            sampling_min=0.035,
            sampling_max=0.08,
        ),
        PDEParameter(
            "anti_diffusion",
            "Destabilizing negative-diffusion coefficient",
            hard_min=0.0,
            sampling_min=0.08,
            sampling_max=0.18,
        ),
        PDEParameter(
            "nonlinear_strength",
            "Nonlinear gradient-saturation strength",
            hard_min=0.0,
            sampling_min=0.35,
            sampling_max=0.8,
        ),
        PDEParameter(
            "damping",
            "Linear damping coefficient",
            hard_min=0.0,
            sampling_min=0.015,
            sampling_max=0.05,
        ),
        PDEParameter(
            "advection_x",
            "Constant x-advection velocity",
            default=0.0,
        ),
        PDEParameter(
            "advection_y",
            "Constant y-advection velocity",
            default=0.0,
        ),
        PDEParameter(
            "advection_z",
            "Constant z-advection velocity",
            default=0.0,
        ),
    ],
    inputs={
        "u": InputSpec(
            name="u",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        )
    },
    boundary_fields={
        "u": BoundaryFieldSpec(
            name="u",
            shape="scalar",
            operators=_KS_BOUNDARY_OPERATORS,
            description="Boundary conditions for the primary field u.",
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
)

__all__ = ["PDE_SPEC"]

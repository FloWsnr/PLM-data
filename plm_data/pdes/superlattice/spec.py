"""Superlattice PDE spec."""

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    SATURATING_STOCHASTIC_COUPLINGS,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

_FIELD_NAMES = ("u_1", "v_1", "u_2", "v_2")

PDE_SPEC = PDESpec(
    name="superlattice",
    category="physics",
    description=(
        "Coupled Brusselator and Lengyel-Epstein reaction-diffusion systems for "
        "superlattice pattern formation."
    ),
    equations={
        "u_1": (
            "du_1/dt = D_u1 * laplacian(u_1) + a - (b + 1) * u_1 "
            "+ u_1^2 * v_1 + alpha * u_1 * u_2 * (u_2 - u_1)"
        ),
        "v_1": "dv_1/dt = D_v1 * laplacian(v_1) + b * u_1 - u_1^2 * v_1",
        "u_2": (
            "du_2/dt = D_u2 * laplacian(u_2) + c - u_2 "
            "- 4 * u_2 * v_2 / (1 + u_2^2) "
            "+ alpha * u_1 * u_2 * (u_1 - u_2)"
        ),
        "v_2": "dv_2/dt = D_v2 * laplacian(v_2) + d * (u_2 - u_2 * v_2 / (1 + u_2^2))",
    },
    parameters=[
        PDEParameter(
            "D_u1",
            "Diffusion coefficient of Brusselator activator u_1",
            hard_min=0.0,
            sampling_min=0.006,
            sampling_max=0.014,
        ),
        PDEParameter(
            "D_v1",
            "Diffusion coefficient of Brusselator inhibitor v_1",
            hard_min=0.0,
            sampling_min=0.05,
            sampling_max=0.12,
        ),
        PDEParameter(
            "D_u2",
            "Diffusion coefficient of Lengyel-Epstein activator u_2",
            hard_min=0.0,
            sampling_min=0.006,
            sampling_max=0.014,
        ),
        PDEParameter(
            "D_v2",
            "Diffusion coefficient of Lengyel-Epstein inhibitor v_2",
            hard_min=0.0,
            sampling_min=0.045,
            sampling_max=0.1,
        ),
        PDEParameter(
            "a",
            "Brusselator kinetic parameter a",
            hard_min=0.0,
            sampling_min=0.9,
            sampling_max=1.1,
        ),
        PDEParameter(
            "b",
            "Brusselator kinetic parameter b",
            hard_min=0.0,
            sampling_min=2.2,
            sampling_max=2.9,
        ),
        PDEParameter(
            "c",
            "Lengyel-Epstein feed parameter c",
            hard_min=0.0,
            sampling_min=4.0,
            sampling_max=5.4,
        ),
        PDEParameter(
            "d",
            "Lengyel-Epstein kinetic parameter d",
            hard_min=0.0,
            sampling_min=7.0,
            sampling_max=10.0,
        ),
        PDEParameter(
            "alpha",
            "Nonlinear coupling strength",
            hard_min=0.0,
            sampling_min=0.02,
            sampling_max=0.08,
        ),
    ],
    inputs={
        field_name: InputSpec(
            name=field_name,
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        )
        for field_name in _FIELD_NAMES
    },
    boundary_fields={
        field_name: BoundaryFieldSpec(
            name=field_name,
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description=f"Boundary conditions for {field_name}.",
        )
        for field_name in _FIELD_NAMES
    },
    states={
        field_name: StateSpec(
            name=field_name,
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        )
        for field_name in _FIELD_NAMES
    },
    outputs={
        field_name: OutputSpec(
            name=field_name,
            shape="scalar",
            output_mode="scalar",
            source_name=field_name,
        )
        for field_name in _FIELD_NAMES
    },
    static_fields=[],
    supported_dimensions=[2],
)

__all__ = ["PDE_SPEC"]

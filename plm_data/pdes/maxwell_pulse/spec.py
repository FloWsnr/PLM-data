"""Maxwell pulse PDE spec."""

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    MAXWELL_BOUNDARY_OPERATORS,
    OutputSpec,
    PDEParameter,
    PDESpec,
    StateSpec,
)

PDE_SPEC = PDESpec(
    name="maxwell_pulse",
    category="physics",
    description=(
        "Transient electric-field pulse propagation using a curl-curl wave "
        "equation with absorbing and PEC boundaries."
    ),
    equations={
        "electric_field": (
            "epsilon_r * d2E/dt2 + sigma * dE/dt + curl(mu_r^-1 curl(E)) = J(t, x)"
        ),
    },
    parameters=[
        PDEParameter(
            "epsilon_r",
            "Relative permittivity",
            hard_min=0.0,
            sampling_min=0.9,
            sampling_max=1.3,
        ),
        PDEParameter(
            "mu_r",
            "Relative permeability",
            hard_min=0.0,
            sampling_min=0.9,
            sampling_max=1.2,
        ),
        PDEParameter(
            "sigma",
            "Ohmic damping coefficient",
            hard_min=0.0,
            sampling_min=0.01,
            sampling_max=0.04,
        ),
        PDEParameter(
            "pulse_amplitude",
            "Pulse amplitude",
            sampling_min=0.5,
            sampling_max=1.0,
        ),
        PDEParameter(
            "pulse_frequency",
            "Pulse carrier frequency",
            hard_min=0.0,
            sampling_min=1.5,
            sampling_max=3.0,
        ),
        PDEParameter(
            "pulse_width",
            "Gaussian pulse width",
            hard_min=0.0,
            sampling_min=0.015,
            sampling_max=0.03,
        ),
        PDEParameter(
            "pulse_delay",
            "Pulse center time",
            sampling_min=0.01,
            sampling_max=0.02,
        ),
    ],
    inputs={
        "electric_field": InputSpec(
            name="electric_field",
            shape="vector",
            allow_source=True,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "electric_field": BoundaryFieldSpec(
            name="electric_field",
            shape="vector",
            operators=MAXWELL_BOUNDARY_OPERATORS,
            description="Boundary conditions for the electric field.",
        )
    },
    states={"electric_field": StateSpec(name="electric_field", shape="vector")},
    outputs={
        "electric_field": OutputSpec(
            name="electric_field",
            shape="vector",
            output_mode="components",
            source_name="electric_field",
        )
    },
    static_fields=[],
    supported_dimensions=[2],
)

__all__ = ["PDE_SPEC"]

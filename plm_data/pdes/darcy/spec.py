"""Darcy PDE spec."""

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    CoefficientSpec,
    GENERIC_STOCHASTIC_COUPLINGS,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

PDE_SPEC = PDESpec(
    name="darcy",
    category="fluids",
    description=(
        "Transient Darcy pressure diffusion in porous media with a passive "
        "tracer transported by the Darcy velocity."
    ),
    equations={
        "pressure": "storage * ∂pressure/∂t = ∇·(mobility ∇pressure) + q_p",
        "velocity": "velocity = -mobility ∇pressure",
        "concentration": (
            "porosity * ∂concentration/∂t + velocity·∇concentration = "
            "∇·(dispersion ∇concentration) + q_c"
        ),
    },
    parameters=[
        PDEParameter(
            "storage",
            "Storage coefficient controlling pressure relaxation",
            hard_min=0.0,
            sampling_min=0.07,
            sampling_max=0.11,
        ),
        PDEParameter(
            "porosity",
            "Pore volume fraction for tracer storage",
            hard_min=0.0,
            hard_max=1.0,
            sampling_min=0.28,
            sampling_max=0.4,
        ),
        PDEParameter(
            "inlet_pressure",
            "Pressure value applied by pressure-drive boundary scenarios.",
            sampling_min=0.9,
            sampling_max=1.2,
        ),
        PDEParameter(
            "outlet_pressure",
            "Outlet pressure value applied by pressure-drive boundary scenarios.",
            sampling_min=-0.04,
            sampling_max=0.08,
        ),
    ],
    inputs={
        "pressure": InputSpec(
            name="pressure",
            shape="scalar",
            allow_source=True,
            allow_initial_condition=True,
        ),
        "concentration": InputSpec(
            name="concentration",
            shape="scalar",
            allow_source=True,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "pressure": BoundaryFieldSpec(
            name="pressure",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the pressure field.",
        ),
        "concentration": BoundaryFieldSpec(
            name="concentration",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the passive tracer field.",
        ),
    },
    states={
        "pressure": StateSpec(
            name="pressure",
            shape="scalar",
            stochastic_couplings=GENERIC_STOCHASTIC_COUPLINGS,
        ),
        "concentration": StateSpec(
            name="concentration",
            shape="scalar",
            stochastic_couplings=GENERIC_STOCHASTIC_COUPLINGS,
        ),
    },
    outputs={
        "pressure": OutputSpec(
            name="pressure",
            shape="scalar",
            output_mode="scalar",
            source_name="pressure",
        ),
        "concentration": OutputSpec(
            name="concentration",
            shape="scalar",
            output_mode="scalar",
            source_name="concentration",
        ),
        "velocity": OutputSpec(
            name="velocity",
            shape="vector",
            output_mode="components",
            source_name="velocity",
            source_kind="derived",
        ),
        "speed": OutputSpec(
            name="speed",
            shape="scalar",
            output_mode="scalar",
            source_name="speed",
            source_kind="derived",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
    coefficients={
        "mobility": CoefficientSpec(
            name="mobility",
            shape="scalar",
            description="Darcy mobility field, e.g. permeability divided by viscosity.",
            allow_randomization=True,
        ),
        "dispersion": CoefficientSpec(
            name="dispersion",
            shape="scalar",
            description="Tracer diffusion/dispersion coefficient field.",
            allow_randomization=True,
        ),
    },
)

__all__ = ["PDE_SPEC"]

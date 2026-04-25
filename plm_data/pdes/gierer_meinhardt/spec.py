"""Gierer meinhardt PDE spec."""

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

PDE_SPEC = PDESpec(
    name="gierer_meinhardt",
    category="biology",
    description="Gierer-Meinhardt activator-inhibitor reaction-diffusion system.",
    equations={
        "a": "da/dt = Da * laplacian(a) + rho_a * a^2 / h - mu_a * a + sigma_a",
        "h": "tau * dh/dt = Dh * laplacian(h) + rho_h * a^2 - mu_h * h + sigma_h",
    },
    parameters=[
        PDEParameter(
            "Da",
            "Diffusion coefficient of activator a",
            hard_min=0.0,
            sampling_min=0.004,
            sampling_max=0.012,
        ),
        PDEParameter(
            "Dh",
            "Diffusion coefficient of inhibitor h",
            hard_min=0.0,
            sampling_min=0.06,
            sampling_max=0.14,
        ),
        PDEParameter(
            "rho_a",
            "Activator self-enhancement rate",
            hard_min=0.0,
            sampling_min=0.12,
            sampling_max=0.28,
        ),
        PDEParameter(
            "rho_h",
            "Cross-activation rate for inhibitor",
            hard_min=0.0,
            sampling_min=0.18,
            sampling_max=0.38,
        ),
        PDEParameter(
            "mu_a",
            "Activator decay rate",
            hard_min=0.0,
            sampling_min=0.35,
            sampling_max=0.65,
        ),
        PDEParameter(
            "mu_h",
            "Inhibitor decay rate",
            hard_min=0.0,
            sampling_min=0.45,
            sampling_max=0.75,
        ),
        PDEParameter(
            "sigma_a",
            "Activator basal production",
            hard_min=0.0,
            sampling_min=0.02,
            sampling_max=0.06,
        ),
        PDEParameter(
            "sigma_h",
            "Inhibitor basal production",
            hard_min=0.0,
            sampling_min=0.02,
            sampling_max=0.06,
        ),
        PDEParameter(
            "tau",
            "Inhibitor time-scale ratio",
            hard_min=0.0,
            sampling_min=0.9,
            sampling_max=1.4,
        ),
    ],
    inputs={
        "a": InputSpec(
            name="a",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "h": InputSpec(
            name="h",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "a": BoundaryFieldSpec(
            name="a",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for activator a.",
        ),
        "h": BoundaryFieldSpec(
            name="h",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for inhibitor h.",
        ),
    },
    states={
        "a": StateSpec(
            name="a",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
        "h": StateSpec(
            name="h",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
    },
    outputs={
        "a": OutputSpec(
            name="a",
            shape="scalar",
            output_mode="scalar",
            source_name="a",
        ),
        "h": OutputSpec(
            name="h",
            shape="scalar",
            output_mode="scalar",
            source_name="h",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
)

__all__ = ["PDE_SPEC"]

"""Immunotherapy PDE spec."""

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
    name="immunotherapy",
    category="biology",
    description=(
        "Three-species cancer immunotherapy model coupling effector cells, "
        "cancer cells, and IL-2 cytokine."
    ),
    equations={
        "u": (
            "du/dt = Du * laplacian(u) + alpha * v - mu_u * u "
            "+ rho_u * u * w / (1 + w) + sigma_u + Ku * t"
        ),
        "v": "dv/dt = Dv * laplacian(v) + v * (1 - v) - u * v / (gamma_v + v)",
        "w": (
            "dw/dt = Dw * laplacian(w) + rho_w * u * v / (gamma_w + v) "
            "- mu_w * w + sigma_w + Kw * t"
        ),
    },
    parameters=[
        PDEParameter(
            "Du",
            "Diffusion coefficient of effector cells",
            hard_min=0.0,
            sampling_min=24.0,
            sampling_max=42.0,
        ),
        PDEParameter(
            "Dv",
            "Diffusion coefficient of cancer cells",
            hard_min=0.0,
            sampling_min=0.55,
            sampling_max=1.0,
        ),
        PDEParameter(
            "Dw",
            "Diffusion coefficient of IL-2 cytokine",
            hard_min=0.0,
            sampling_min=18.0,
            sampling_max=34.0,
        ),
        PDEParameter(
            "alpha",
            "Effector recruitment rate per cancer density",
            hard_min=0.0,
            sampling_min=0.045,
            sampling_max=0.085,
        ),
        PDEParameter(
            "mu_u",
            "Natural death rate of effector cells",
            hard_min=0.0,
            sampling_min=0.16,
            sampling_max=0.24,
        ),
        PDEParameter(
            "rho_u",
            "IL-2-stimulated proliferation rate of effectors",
            hard_min=0.0,
            sampling_min=0.38,
            sampling_max=0.62,
        ),
        PDEParameter(
            "gamma_v",
            "Cancer-density half-saturation for immune killing",
            hard_min=0.0,
            sampling_min=0.12,
            sampling_max=0.22,
        ),
        PDEParameter(
            "rho_w",
            "IL-2 production rate from effector-cancer interaction",
            hard_min=0.0,
            sampling_min=1.2,
            sampling_max=2.1,
        ),
        PDEParameter(
            "gamma_w",
            "Cancer-density half-saturation for IL-2 production",
            hard_min=0.0,
            sampling_min=0.0015,
            sampling_max=0.0045,
        ),
        PDEParameter(
            "mu_w",
            "Natural degradation rate of IL-2",
            hard_min=0.0,
            sampling_min=24.0,
            sampling_max=40.0,
        ),
        PDEParameter(
            "sigma_u",
            "Basal effector cell infusion rate",
            hard_min=0.0,
            sampling_min=0.0,
            sampling_max=0.006,
        ),
        PDEParameter(
            "Ku",
            "Linear-in-time effector cell treatment rate",
            hard_min=0.0,
            sampling_min=0.0,
            sampling_max=0.00004,
        ),
        PDEParameter(
            "sigma_w",
            "Basal IL-2 infusion rate",
            hard_min=0.0,
            sampling_min=0.0,
            sampling_max=0.003,
        ),
        PDEParameter(
            "Kw",
            "Linear-in-time IL-2 treatment rate",
            hard_min=0.0,
            sampling_min=0.0,
            sampling_max=0.00002,
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
        "w": InputSpec(
            name="w",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "u": BoundaryFieldSpec(
            name="u",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for effector cells u.",
        ),
        "v": BoundaryFieldSpec(
            name="v",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for cancer cells v.",
        ),
        "w": BoundaryFieldSpec(
            name="w",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for IL-2 cytokine w.",
        ),
    },
    states={
        "u": StateSpec(
            name="u",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
        "v": StateSpec(
            name="v",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
        "w": StateSpec(
            name="w",
            shape="scalar",
            stochastic_couplings=SATURATING_STOCHASTIC_COUPLINGS,
        ),
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
        "w": OutputSpec(
            name="w",
            shape="scalar",
            output_mode="scalar",
            source_name="w",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
)

__all__ = ["PDE_SPEC"]

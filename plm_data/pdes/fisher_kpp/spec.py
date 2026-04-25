"""Fisher kpp PDE spec."""

from plm_data.pdes.fisher_kpp.helpers import build_scalar_reaction_diffusion_spec
from plm_data.pdes.metadata import PDEParameter

PDE_SPEC = build_scalar_reaction_diffusion_spec(
    name="fisher_kpp",
    description=(
        "Fisher-KPP logistic reaction-diffusion equation for invasion fronts "
        "and travelling waves."
    ),
    reaction_equation="r * u * (1 - u / K)",
    diffusion_parameter=PDEParameter(
        "D",
        "Diffusion coefficient",
        hard_min=0.0,
        sampling_min=0.03,
        sampling_max=0.08,
    ),
    parameters=[
        PDEParameter(
            "r",
            "Intrinsic growth rate",
            hard_min=0.0,
            sampling_min=0.45,
            sampling_max=0.9,
        ),
        PDEParameter(
            "K",
            "Carrying capacity",
            hard_min=0.0,
            sampling_min=0.9,
            sampling_max=1.2,
        ),
    ],
)

__all__ = ["PDE_SPEC"]

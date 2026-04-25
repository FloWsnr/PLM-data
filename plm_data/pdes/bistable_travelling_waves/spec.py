"""Bistable travelling waves PDE spec."""

from plm_data.pdes.fisher_kpp.helpers import build_scalar_reaction_diffusion_spec
from plm_data.pdes.metadata import PDEParameter

PDE_SPEC = build_scalar_reaction_diffusion_spec(
    name="bistable_travelling_waves",
    description=(
        "Bistable reaction-diffusion equation with Allen-Cahn / "
        "Chaffee-Infante kinetics for travelling fronts."
    ),
    reaction_equation="u * (u - a) * (1 - u)",
    diffusion_parameter=PDEParameter(
        "D",
        "Diffusion coefficient",
        hard_min=0.0,
        sampling_min=0.025,
        sampling_max=0.06,
    ),
    parameters=[
        PDEParameter(
            "a",
            "Bistability threshold parameter",
            hard_min=0.0,
            hard_max=1.0,
            sampling_min=0.25,
            sampling_max=0.45,
        ),
    ],
)

__all__ = ["PDE_SPEC"]

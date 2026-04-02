"""Fisher-KPP reaction-diffusion preset."""

from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance
from plm_data.presets.metadata import PDEParameter

from ._scalar_reaction_diffusion import (
    ScalarReactionDiffusionProblem,
    build_scalar_reaction_diffusion_spec,
)

_FISHER_KPP_SPEC = build_scalar_reaction_diffusion_spec(
    name="fisher_kpp",
    description=(
        "Fisher-KPP logistic reaction-diffusion equation for invasion fronts "
        "and travelling waves."
    ),
    reaction_equation="r * u * (1 - u / K)",
    parameters=[
        PDEParameter("r", "Intrinsic growth rate"),
        PDEParameter("K", "Carrying capacity"),
    ],
)


class _FisherKPPProblem(ScalarReactionDiffusionProblem):
    def reaction_term(self, u_current):
        r = self.config.parameters["r"]
        K = self.config.parameters["K"]
        return r * u_current * (1.0 - u_current / K)


@register_preset("fisher_kpp")
class FisherKPPPreset(PDEPreset):
    @property
    def spec(self):
        return _FISHER_KPP_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _FisherKPPProblem(self.spec, config)

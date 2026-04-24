"""Fisher-KPP reaction-diffusion PDE."""

from plm_data.pdes.base import PDE, ProblemInstance
from plm_data.pdes.metadata import PDEParameter

from plm_data.pdes.fisher_kpp.helpers import (
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


class FisherKPPPDE(PDE):
    @property
    def spec(self):
        return _FISHER_KPP_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _FisherKPPProblem(self.spec, config)

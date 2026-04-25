"""Fisher-KPP reaction-diffusion PDE."""

from plm_data.pdes.base import PDE, ProblemInstance

from plm_data.pdes.fisher_kpp.helpers import (
    ScalarReactionDiffusionProblem,
)
from plm_data.pdes.fisher_kpp.spec import PDE_SPEC


class _FisherKPPProblem(ScalarReactionDiffusionProblem):
    def reaction_term(self, u_current):
        r = self.config.parameters["r"]
        K = self.config.parameters["K"]
        return r * u_current * (1.0 - u_current / K)


class FisherKPPPDE(PDE):
    @property
    def spec(self):
        return PDE_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _FisherKPPProblem(self.spec, config)

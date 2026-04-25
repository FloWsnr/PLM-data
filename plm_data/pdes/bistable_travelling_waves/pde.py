"""Bistable Allen-Cahn / Chaffee-Infante travelling-wave PDE."""

from plm_data.pdes.base import PDE, ProblemInstance

from plm_data.pdes.fisher_kpp.helpers import (
    ScalarReactionDiffusionProblem,
)
from plm_data.pdes.bistable_travelling_waves.spec import PDE_SPEC


class _BistableTravellingWavesProblem(ScalarReactionDiffusionProblem):
    def reaction_term(self, u_current):
        a = self.config.parameters["a"]
        return u_current * (u_current - a) * (1.0 - u_current)


class BistableTravellingWavesPDE(PDE):
    @property
    def spec(self):
        return PDE_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _BistableTravellingWavesProblem(self.spec, config)

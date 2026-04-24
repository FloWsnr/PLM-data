"""Bistable Allen-Cahn / Chaffee-Infante travelling-wave PDE."""

from plm_data.pdes.base import PDE, ProblemInstance
from plm_data.pdes.metadata import PDEParameter

from plm_data.pdes.fisher_kpp.helpers import (
    ScalarReactionDiffusionProblem,
    build_scalar_reaction_diffusion_spec,
)

_BISTABLE_TRAVELLING_WAVES_SPEC = build_scalar_reaction_diffusion_spec(
    name="bistable_travelling_waves",
    description=(
        "Bistable reaction-diffusion equation with Allen-Cahn / "
        "Chaffee-Infante kinetics for travelling fronts."
    ),
    reaction_equation="u * (u - a) * (1 - u)",
    parameters=[
        PDEParameter("a", "Bistability threshold parameter"),
    ],
)


class _BistableTravellingWavesProblem(ScalarReactionDiffusionProblem):
    def reaction_term(self, u_current):
        a = self.config.parameters["a"]
        return u_current * (u_current - a) * (1.0 - u_current)


class BistableTravellingWavesPDE(PDE):
    @property
    def spec(self):
        return _BISTABLE_TRAVELLING_WAVES_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _BistableTravellingWavesProblem(self.spec, config)

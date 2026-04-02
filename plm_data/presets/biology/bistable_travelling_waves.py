"""Bistable Allen-Cahn / Chaffee-Infante travelling-wave preset."""

from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance
from plm_data.presets.metadata import PDEParameter

from ._scalar_reaction_diffusion import (
    ScalarReactionDiffusionProblem,
    build_scalar_reaction_diffusion_spec,
)

_BISTABLE_TRAVELLING_WAVES_SPEC = build_scalar_reaction_diffusion_spec(
    name="bistable_travelling_waves",
    description=(
        "Bistable reaction-diffusion equation with Allen-Cahn / "
        "Chaffee-Infante kinetics for travelling fronts and Allee-like "
        "persistence thresholds."
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


@register_preset("bistable_travelling_waves")
class BistableTravellingWavesPreset(PDEPreset):
    @property
    def spec(self):
        return _BISTABLE_TRAVELLING_WAVES_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _BistableTravellingWavesProblem(self.spec, config)

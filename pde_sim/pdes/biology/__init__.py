"""Mathematical Biology PDEs: Schnakenberg, FitzHugh-Nagumo, etc."""

from .schnakenberg import SchnakenbergPDE, BrusselatorPDE, FisherKPPPDE
from .fitzhugh_nagumo import FitzHughNagumoPDE, AllenCahnPDE
from .gierer_meinhardt import GiererMeinhardtPDE, KellerSegelPDE
from .advanced import (
    CyclicCompetitionPDE,
    VegetationPDE,
    CrossDiffusionPDE,
    ImmunotherapyPDE,
    HarshEnvironmentPDE,
    BacteriaFlowPDE,
    HeterogeneousPDE,
    TopographyPDE,
    TuringConditionsPDE,
)

__all__ = [
    "SchnakenbergPDE",
    "BrusselatorPDE",
    "FisherKPPPDE",
    "FitzHughNagumoPDE",
    "AllenCahnPDE",
    "GiererMeinhardtPDE",
    "KellerSegelPDE",
    "CyclicCompetitionPDE",
    "VegetationPDE",
    "CrossDiffusionPDE",
    "ImmunotherapyPDE",
    "HarshEnvironmentPDE",
    "BacteriaFlowPDE",
    "HeterogeneousPDE",
    "TopographyPDE",
    "TuringConditionsPDE",
]

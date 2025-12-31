"""Mathematical Biology PDEs: Schnakenberg, FitzHugh-Nagumo, etc."""

from .schnakenberg import SchnakenbergPDE, BrusselatorPDE, FisherKPPPDE
from .fitzhugh_nagumo import FitzHughNagumoPDE, AllenCahnPDE, StandardAllenCahnPDE
from .gierer_meinhardt import GiererMeinhardtPDE, KellerSegelPDE
from .sir import SIRModelPDE
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
    "StandardAllenCahnPDE",
    "GiererMeinhardtPDE",
    "KellerSegelPDE",
    "SIRModelPDE",
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

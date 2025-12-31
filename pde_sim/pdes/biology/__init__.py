"""Mathematical Biology PDEs: Schnakenberg, FitzHugh-Nagumo, etc."""

from .schnakenberg import SchnakenbergPDE
from .brusselator import BrusselatorPDE
from .fisher_kpp import FisherKPPPDE
from .fitzhugh_nagumo import FitzHughNagumoPDE
from .allen_cahn import AllenCahnPDE, StandardAllenCahnPDE
from .gierer_meinhardt import GiererMeinhardtPDE
from .keller_segel import KellerSegelPDE
from .sir import SIRModelPDE
from .cyclic_competition import CyclicCompetitionPDE
from .vegetation import VegetationPDE
from .cross_diffusion import CrossDiffusionPDE
from .immunotherapy import ImmunotherapyPDE
from .harsh_environment import HarshEnvironmentPDE
from .bacteria_flow import BacteriaFlowPDE
from .heterogeneous import HeterogeneousPDE
from .topography import TopographyPDE
from .turing_conditions import TuringConditionsPDE

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

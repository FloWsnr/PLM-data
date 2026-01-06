"""Mathematical Biology PDEs: Schnakenberg, FitzHugh-Nagumo, etc."""

from .schnakenberg import SchnakenbergPDE
from .cross_diffusion_schnakenberg import CrossDiffusionSchnakenbergPDE
from .brusselator import BrusselatorPDE
from .hyperbolic_brusselator import HyperbolicBrusselatorPDE
from .fisher_kpp import FisherKPPPDE
from .fitzhugh_nagumo import FitzHughNagumoPDE
from .allen_cahn import BistableAllenCahnPDE
from .gierer_meinhardt import GiererMeinhardtPDE
from .heterogeneous_gierer_meinhardt import HeterogeneousGiererMeinhardtPDE
from .keller_segel import KellerSegelPDE
from .klausmeier import KlausmeierPDE
from .klausmeier_topography import KlausmeierTopographyPDE
from .cyclic_competition import CyclicCompetitionPDE
from .immunotherapy import ImmunotherapyPDE
from .harsh_environment import HarshEnvironmentPDE
from .bacteria_advection import BacteriaAdvectionPDE

__all__ = [
    "SchnakenbergPDE",
    "CrossDiffusionSchnakenbergPDE",
    "BrusselatorPDE",
    "HyperbolicBrusselatorPDE",
    "FisherKPPPDE",
    "FitzHughNagumoPDE",
    "BistableAllenCahnPDE",
    "GiererMeinhardtPDE",
    "HeterogeneousGiererMeinhardtPDE",
    "KellerSegelPDE",
    "KlausmeierPDE",
    "KlausmeierTopographyPDE",
    "CyclicCompetitionPDE",
    "ImmunotherapyPDE",
    "HarshEnvironmentPDE",
    "BacteriaAdvectionPDE",
]

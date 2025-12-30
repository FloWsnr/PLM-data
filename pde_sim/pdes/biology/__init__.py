"""Mathematical Biology PDEs: Schnakenberg, FitzHugh-Nagumo, etc."""

from .schnakenberg import SchnakenbergPDE, BrusselatorPDE, FisherKPPPDE
from .fitzhugh_nagumo import FitzHughNagumoPDE, AllenCahnPDE

__all__ = [
    "SchnakenbergPDE",
    "BrusselatorPDE",
    "FisherKPPPDE",
    "FitzHughNagumoPDE",
    "AllenCahnPDE",
]

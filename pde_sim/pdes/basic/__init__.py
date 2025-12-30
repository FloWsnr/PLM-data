"""Basic PDEs: Heat, Wave, Advection, etc."""

from .heat import HeatPDE, InhomogeneousHeatPDE
from .wave import WavePDE, AdvectionPDE

__all__ = [
    "HeatPDE",
    "InhomogeneousHeatPDE",
    "WavePDE",
    "AdvectionPDE",
]

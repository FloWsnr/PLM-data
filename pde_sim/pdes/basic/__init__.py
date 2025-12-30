"""Basic PDEs: Heat, Wave, Advection, etc."""

from .heat import HeatPDE, InhomogeneousHeatPDE
from .wave import WavePDE, AdvectionPDE
from .schrodinger import SchrodingerPDE, PlatePDE

__all__ = [
    "HeatPDE",
    "InhomogeneousHeatPDE",
    "WavePDE",
    "AdvectionPDE",
    "SchrodingerPDE",
    "PlatePDE",
]

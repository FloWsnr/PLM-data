"""Basic PDEs: Heat, Wave, Advection, Schrodinger, Plate, etc."""

from .heat import HeatPDE, InhomogeneousHeatPDE, InhomogeneousDiffusionHeatPDE
from .wave import WavePDE, InhomogeneousWavePDE
from .advection import AdvectionPDE
from .schrodinger import SchrodingerPDE
from .plate import PlatePDE

__all__ = [
    "HeatPDE",
    "InhomogeneousHeatPDE",
    "InhomogeneousDiffusionHeatPDE",
    "WavePDE",
    "InhomogeneousWavePDE",
    "AdvectionPDE",
    "SchrodingerPDE",
    "PlatePDE",
]

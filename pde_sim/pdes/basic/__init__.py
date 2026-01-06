"""Basic PDEs: Heat, Wave, Advection, Schrodinger, Plate, etc."""

from .heat import HeatPDE, InhomogeneousHeatPDE
from .wave import WavePDE, AdvectionPDE, InhomogeneousWavePDE
from .schrodinger import SchrodingerPDE, PlatePDE

__all__ = [
    "HeatPDE",
    "InhomogeneousHeatPDE",
    "WavePDE",
    "InhomogeneousWavePDE",
    "AdvectionPDE",
    "SchrodingerPDE",
    "PlatePDE",
]

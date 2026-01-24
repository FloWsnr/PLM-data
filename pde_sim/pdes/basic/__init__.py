"""Basic PDEs: Heat, Wave, Advection, Schrodinger, Plate, etc."""

from .heat import HeatPDE, InhomogeneousHeatPDE, InhomogeneousDiffusionHeatPDE
from .wave import WavePDE, InhomogeneousWavePDE
from .wave_standing import StandingWavePDE
from .damped_wave import DampedWavePDE
from .advection import AdvectionPDE
from .advection_rotational import AdvectionRotationalPDE
from .schrodinger import SchrodingerPDE
from .plate import PlatePDE

__all__ = [
    "HeatPDE",
    "InhomogeneousHeatPDE",
    "InhomogeneousDiffusionHeatPDE",
    "WavePDE",
    "InhomogeneousWavePDE",
    "StandingWavePDE",
    "DampedWavePDE",
    "AdvectionPDE",
    "AdvectionRotationalPDE",
    "SchrodingerPDE",
    "PlatePDE",
]

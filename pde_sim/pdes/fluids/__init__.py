"""Fluid Dynamics PDEs: Navier-Stokes, Shallow Water, etc."""

from .vorticity import VorticityPDE
from .shallow_water import ShallowWaterPDE
from .navier_stokes import NavierStokesPDE
from .thermal_convection import ThermalConvectionPDE
from .potential_flow_dipoles import PotentialFlowDipolesPDE
from .potential_flow_images import PotentialFlowImagesPDE

__all__ = [
    "VorticityPDE",
    "ShallowWaterPDE",
    "NavierStokesPDE",
    "ThermalConvectionPDE",
    "PotentialFlowDipolesPDE",
    "PotentialFlowImagesPDE",
]

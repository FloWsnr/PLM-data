"""Fluid Dynamics PDEs: Navier-Stokes, Shallow Water, etc."""

from .darcy import DarcyPDE
from .vorticity import VorticityPDE
from .vorticity_bounded import VorticityBoundedPDE
from .shallow_water import ShallowWaterPDE
from .navier_stokes import NavierStokesPDE
from .navier_stokes_cylinder import NavierStokesCylinderPDE
from .thermal_convection import ThermalConvectionPDE
from .potential_flow_dipoles import PotentialFlowDipolesPDE

__all__ = [
    "DarcyPDE",
    "VorticityPDE",
    "VorticityBoundedPDE",
    "ShallowWaterPDE",
    "NavierStokesPDE",
    "NavierStokesCylinderPDE",
    "ThermalConvectionPDE",
    "PotentialFlowDipolesPDE",
]

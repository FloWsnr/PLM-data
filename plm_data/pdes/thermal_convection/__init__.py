"""Thermal-convection PDE package."""

from plm_data.pdes.thermal_convection.pde import ThermalConvectionPDE
from plm_data.pdes.thermal_convection.spec import PDE_SPEC

__all__ = ["PDE_SPEC", "ThermalConvectionPDE"]

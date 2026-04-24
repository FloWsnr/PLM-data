"""Thermal-convection PDE spec."""

from plm_data.pdes.thermal_convection.pde import ThermalConvectionPDE

PDE_SPEC = ThermalConvectionPDE().spec

__all__ = ["PDE_SPEC"]

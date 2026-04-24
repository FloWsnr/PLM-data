"""Heat PDE spec."""

from plm_data.pdes.heat.pde import HeatPDE

PDE_SPEC = HeatPDE().spec

__all__ = ["PDE_SPEC"]

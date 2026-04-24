"""Van der Pol PDE spec."""

from plm_data.pdes.van_der_pol.pde import VanDerPolPDE

PDE_SPEC = VanDerPolPDE().spec

__all__ = ["PDE_SPEC"]

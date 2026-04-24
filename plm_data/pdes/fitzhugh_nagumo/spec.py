"""FitzHugh-Nagumo PDE spec."""

from plm_data.pdes.fitzhugh_nagumo.pde import FitzHughNagumoPDE

PDE_SPEC = FitzHughNagumoPDE().spec

__all__ = ["PDE_SPEC"]

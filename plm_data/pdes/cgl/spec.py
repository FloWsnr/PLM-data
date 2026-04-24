"""Complex Ginzburg-Landau PDE spec."""

from plm_data.pdes.cgl.pde import CGLPDE

PDE_SPEC = CGLPDE().spec

__all__ = ["PDE_SPEC"]

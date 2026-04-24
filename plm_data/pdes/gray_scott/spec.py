"""Gray-Scott PDE spec."""

from plm_data.pdes.gray_scott.pde import GrayScottPDE

PDE_SPEC = GrayScottPDE().spec

__all__ = ["PDE_SPEC"]

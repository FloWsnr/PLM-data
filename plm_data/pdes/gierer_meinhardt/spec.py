"""Gierer-Meinhardt PDE spec."""

from plm_data.pdes.gierer_meinhardt.pde import GiererMeinhardtPDE

PDE_SPEC = GiererMeinhardtPDE().spec

__all__ = ["PDE_SPEC"]

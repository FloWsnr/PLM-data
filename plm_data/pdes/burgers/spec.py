"""Burgers PDE spec."""

from plm_data.pdes.burgers.pde import BurgersPDE

PDE_SPEC = BurgersPDE().spec

__all__ = ["PDE_SPEC"]

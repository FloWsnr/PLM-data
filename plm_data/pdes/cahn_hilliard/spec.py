"""Cahn-Hilliard PDE spec."""

from plm_data.pdes.cahn_hilliard.pde import CahnHilliardPDE

PDE_SPEC = CahnHilliardPDE().spec

__all__ = ["PDE_SPEC"]

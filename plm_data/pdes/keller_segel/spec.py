"""Keller-Segel PDE spec."""

from plm_data.pdes.keller_segel.pde import KellerSegelPDE

PDE_SPEC = KellerSegelPDE().spec

__all__ = ["PDE_SPEC"]

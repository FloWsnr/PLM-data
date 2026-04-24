"""Zakharov-Kuznetsov PDE spec."""

from plm_data.pdes.zakharov_kuznetsov.pde import ZakharovKuznetsovPDE

PDE_SPEC = ZakharovKuznetsovPDE().spec

__all__ = ["PDE_SPEC"]

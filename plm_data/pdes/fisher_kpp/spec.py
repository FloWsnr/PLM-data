"""Fisher-KPP PDE spec."""

from plm_data.pdes.fisher_kpp.pde import FisherKPPPDE

PDE_SPEC = FisherKPPPDE().spec

__all__ = ["PDE_SPEC"]

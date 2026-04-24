"""Cyclic competition PDE spec."""

from plm_data.pdes.cyclic_competition.pde import CyclicCompetitionPDE

PDE_SPEC = CyclicCompetitionPDE().spec

__all__ = ["PDE_SPEC"]

"""Bistable travelling-waves PDE spec."""

from plm_data.pdes.bistable_travelling_waves.pde import BistableTravellingWavesPDE

PDE_SPEC = BistableTravellingWavesPDE().spec

__all__ = ["PDE_SPEC"]

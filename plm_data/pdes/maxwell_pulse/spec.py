"""Maxwell pulse PDE spec."""

from plm_data.pdes.maxwell_pulse.pde import MaxwellPulsePDE

PDE_SPEC = MaxwellPulsePDE().spec

__all__ = ["PDE_SPEC"]

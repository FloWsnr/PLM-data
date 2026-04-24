"""Wave PDE spec."""

from plm_data.pdes.wave.pde import WavePDE

PDE_SPEC = WavePDE().spec

__all__ = ["PDE_SPEC"]

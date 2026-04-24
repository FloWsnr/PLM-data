"""Compatibility imports for airfoil helpers now owned by domains."""

from plm_data.domains.airfoil import (
    symmetric_naca_airfoil_outline,
    symmetric_naca_airfoil_surfaces,
)

__all__ = [
    "symmetric_naca_airfoil_outline",
    "symmetric_naca_airfoil_surfaces",
]

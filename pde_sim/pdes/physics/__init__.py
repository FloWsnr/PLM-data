"""Nonlinear Physics PDEs: Gray-Scott, Swift-Hohenberg, etc."""

from .gray_scott import GrayScottPDE
from .cahn_hilliard import CahnHilliardPDE, SwiftHohenbergPDE, BurgersPDE

__all__ = [
    "GrayScottPDE",
    "CahnHilliardPDE",
    "SwiftHohenbergPDE",
    "BurgersPDE",
]

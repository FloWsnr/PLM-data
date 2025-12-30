"""Nonlinear Physics PDEs: Gray-Scott, Swift-Hohenberg, etc."""

from .gray_scott import GrayScottPDE
from .cahn_hilliard import CahnHilliardPDE, SwiftHohenbergPDE, BurgersPDE
from .kuramoto_sivashinsky import KuramotoSivashinskyPDE, KdVPDE, GinzburgLandauPDE

__all__ = [
    "GrayScottPDE",
    "CahnHilliardPDE",
    "SwiftHohenbergPDE",
    "BurgersPDE",
    "KuramotoSivashinskyPDE",
    "KdVPDE",
    "GinzburgLandauPDE",
]

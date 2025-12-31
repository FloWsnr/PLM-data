"""Nonlinear Physics PDEs: Gray-Scott, Swift-Hohenberg, etc."""

from .gray_scott import GrayScottPDE
from .cahn_hilliard import CahnHilliardPDE
from .swift_hohenberg import SwiftHohenbergPDE
from .burgers import BurgersPDE
from .kuramoto_sivashinsky import KuramotoSivashinskyPDE
from .kdv import KdVPDE
from .ginzburg_landau import GinzburgLandauPDE
from .kpz import KPZInterfacePDE
from .lorenz import LorenzPDE
from .superlattice import SuperlatticePDE
from .oscillators import OscillatorsPDE
from .perona_malik import PeronaMalikPDE
from .nonlinear_beams import NonlinearBeamsPDE
from .turing_wave import TuringWavePDE
from .advecting_patterns import AdvectingPatternsPDE
from .growing_domains import GrowingDomainsPDE

__all__ = [
    "GrayScottPDE",
    "CahnHilliardPDE",
    "SwiftHohenbergPDE",
    "BurgersPDE",
    "KuramotoSivashinskyPDE",
    "KdVPDE",
    "GinzburgLandauPDE",
    "KPZInterfacePDE",
    "LorenzPDE",
    "SuperlatticePDE",
    "OscillatorsPDE",
    "PeronaMalikPDE",
    "NonlinearBeamsPDE",
    "TuringWavePDE",
    "AdvectingPatternsPDE",
    "GrowingDomainsPDE",
]

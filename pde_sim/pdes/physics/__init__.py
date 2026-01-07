"""Physics PDEs - pattern formation, waves, chaos, and nonlinear dynamics."""

# Basic pattern formation
from .gray_scott import GrayScottPDE
from .bistable_advection import BistableAdvectionPDE
from .stochastic_gray_scott import StochasticGrayScottPDE
from .cahn_hilliard import CahnHilliardPDE
from .swift_hohenberg import SwiftHohenbergPDE
from .swift_hohenberg_advection import SwiftHohenbergAdvectionPDE
from .superlattice import SuperlatticePDE

# Wave and soliton equations
from .burgers import BurgersPDE
from .inviscid_burgers import InviscidBurgersPDE
from .korteweg_de_vries import KortewegDeVriesPDE
from .kuramoto_sivashinsky import KuramotoSivashinskyPDE
from .nonlinear_schrodinger import NonlinearSchrodingerPDE
from .complex_ginzburg_landau import ComplexGinzburgLandauPDE

# Chaotic systems
from .lorenz import LorenzPDE

# Image processing and diffusion
from .perona_malik import PeronaMalikPDE

# Structural mechanics
from .nonlinear_beam import NonlinearBeamPDE

# Nonlinear oscillators
from .van_der_pol import VanDerPolPDE
from .duffing import DuffingPDE

__all__ = [
    # Pattern formation
    "GrayScottPDE",
    "BistableAdvectionPDE",
    "StochasticGrayScottPDE",
    "CahnHilliardPDE",
    "SwiftHohenbergPDE",
    "SwiftHohenbergAdvectionPDE",
    "SuperlatticePDE",
    # Waves and solitons
    "BurgersPDE",
    "InviscidBurgersPDE",
    "KortewegDeVriesPDE",
    "KuramotoSivashinskyPDE",
    "NonlinearSchrodingerPDE",
    "ComplexGinzburgLandauPDE",
    # Chaos
    "LorenzPDE",
    # Image processing
    "PeronaMalikPDE",
    # Structural mechanics
    "NonlinearBeamPDE",
    # Oscillators
    "VanDerPolPDE",
    "DuffingPDE",
]

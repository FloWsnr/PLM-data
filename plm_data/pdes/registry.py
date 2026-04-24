"""PDE registry using the refactored naming vocabulary."""

from collections.abc import Callable

from plm_data.pdes.advection import AdvectionPDE
from plm_data.pdes.bistable_travelling_waves import BistableTravellingWavesPDE
from plm_data.pdes.brusselator import BrusselatorPDE
from plm_data.pdes.burgers import BurgersPDE
from plm_data.pdes.cahn_hilliard import CahnHilliardPDE
from plm_data.pdes.cgl import CGLPDE
from plm_data.pdes.cyclic_competition import CyclicCompetitionPDE
from plm_data.pdes.darcy import DarcyPDE
from plm_data.pdes.elasticity import ElasticityPDE
from plm_data.pdes.fisher_kpp import FisherKPPPDE
from plm_data.pdes.fitzhugh_nagumo import FitzHughNagumoPDE
from plm_data.pdes.gierer_meinhardt import GiererMeinhardtPDE
from plm_data.pdes.gray_scott import GrayScottPDE
from plm_data.pdes.heat import HeatPDE
from plm_data.pdes.immunotherapy import ImmunotherapyPDE
from plm_data.pdes.keller_segel import KellerSegelPDE
from plm_data.pdes.klausmeier_topography import KlausmeierTopographyPDE
from plm_data.pdes.kuramoto_sivashinsky import KuramotoSivashinskyPDE
from plm_data.pdes.lorenz import LorenzPDE
from plm_data.pdes.maxwell_pulse import MaxwellPulsePDE
from plm_data.pdes.navier_stokes import NavierStokesPDE
from plm_data.pdes.plate import PlatePDE
from plm_data.pdes.schrodinger import SchrodingerPDE
from plm_data.pdes.schnakenberg import SchnakenbergPDE
from plm_data.pdes.shallow_water import ShallowWaterPDE
from plm_data.pdes.superlattice import SuperlatticePDE
from plm_data.pdes.swift_hohenberg import SwiftHohenbergPDE
from plm_data.pdes.thermal_convection import ThermalConvectionPDE
from plm_data.pdes.van_der_pol import VanDerPolPDE
from plm_data.pdes.wave import WavePDE
from plm_data.pdes.zakharov_kuznetsov import ZakharovKuznetsovPDE
from plm_data.pdes.base import PDE

_PDE_REGISTRY: dict[str, type[PDE]] = {
    "advection": AdvectionPDE,
    "wave": WavePDE,
    "plate": PlatePDE,
    "schrodinger": SchrodingerPDE,
    "burgers": BurgersPDE,
    "heat": HeatPDE,
    "bistable_travelling_waves": BistableTravellingWavesPDE,
    "brusselator": BrusselatorPDE,
    "fitzhugh_nagumo": FitzHughNagumoPDE,
    "gray_scott": GrayScottPDE,
    "gierer_meinhardt": GiererMeinhardtPDE,
    "schnakenberg": SchnakenbergPDE,
    "cyclic_competition": CyclicCompetitionPDE,
    "immunotherapy": ImmunotherapyPDE,
    "van_der_pol": VanDerPolPDE,
    "lorenz": LorenzPDE,
    "keller_segel": KellerSegelPDE,
    "klausmeier_topography": KlausmeierTopographyPDE,
    "superlattice": SuperlatticePDE,
    "cahn_hilliard": CahnHilliardPDE,
    "cgl": CGLPDE,
    "kuramoto_sivashinsky": KuramotoSivashinskyPDE,
    "swift_hohenberg": SwiftHohenbergPDE,
    "zakharov_kuznetsov": ZakharovKuznetsovPDE,
    "elasticity": ElasticityPDE,
    "fisher_kpp": FisherKPPPDE,
    "darcy": DarcyPDE,
    "navier_stokes": NavierStokesPDE,
    "shallow_water": ShallowWaterPDE,
    "thermal_convection": ThermalConvectionPDE,
    "maxwell_pulse": MaxwellPulsePDE,
}


def register_pde(name: str) -> Callable[[type[PDE]], type[PDE]]:
    """Decorator to register a PDE implementation class."""

    def decorator(cls: type[PDE]) -> type[PDE]:
        _PDE_REGISTRY[name] = cls
        return cls

    return decorator


def get_pde(name: str) -> PDE:
    """Instantiate a registered PDE by name."""
    if name not in _PDE_REGISTRY:
        available = ", ".join(sorted(_PDE_REGISTRY))
        raise ValueError(f"Unknown migrated PDE '{name}'. Available: {available}")
    return _PDE_REGISTRY[name]()


def list_pdes() -> dict[str, type[PDE]]:
    """Return the PDE implementation classes migrated to the random runner."""
    return dict(_PDE_REGISTRY)


__all__ = [
    "get_pde",
    "list_pdes",
    "register_pde",
]

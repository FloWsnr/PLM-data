"""Cahn-Hilliard equation for phase separation."""

from typing import Any

from pde import PDE, CartesianGrid

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("cahn-hilliard")
class CahnHilliardPDE(ScalarPDEPreset):
    """Cahn-Hilliard equation for phase separation.

    The Cahn-Hilliard equation describes phase separation in binary mixtures:

        du/dt = M * laplace(u³ - u - gamma * laplace(u))

    where:
        - u is the concentration difference between phases (-1 to 1)
        - M is mobility
        - gamma controls interface width

    The equation conserves the total amount of u.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="cahn-hilliard",
            category="physics",
            description="Cahn-Hilliard phase separation",
            equations={
                "u": "M * laplace(u**3 - u - gamma * laplace(u))",
            },
            parameters=[
                PDEParameter(
                    name="M",
                    default=1.0,
                    description="Mobility coefficient",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="gamma",
                    default=0.01,
                    description="Interface width parameter",
                    min_value=0.001,
                    max_value=1.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Phase separation in binary mixtures",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        M = parameters.get("M", 1.0)
        gamma = parameters.get("gamma", 0.01)

        return PDE(
            rhs={"u": f"{M} * laplace(u**3 - u - {gamma} * laplace(u))"},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )


@register_pde("swift-hohenberg")
class SwiftHohenbergPDE(ScalarPDEPreset):
    """Swift-Hohenberg equation for pattern formation.

    The Swift-Hohenberg equation is a model for pattern formation:

        du/dt = r*u - (k²+laplace)²*u + g₂*u² + g₃*u³

    where:
        - r is the control parameter
        - k is the critical wavenumber
        - g₂, g₃ are nonlinear coefficients
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="swift-hohenberg",
            category="physics",
            description="Swift-Hohenberg pattern formation",
            equations={
                "u": "r*u - (k**2 + laplace(u))**2 + g2*u**2 + g3*u**3",
            },
            parameters=[
                PDEParameter(
                    name="r",
                    default=0.1,
                    description="Control parameter",
                    min_value=-1.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="k",
                    default=1.0,
                    description="Critical wavenumber",
                    min_value=0.1,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="g2",
                    default=0.0,
                    description="Quadratic nonlinearity",
                    min_value=-5.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="g3",
                    default=-1.0,
                    description="Cubic nonlinearity (should be negative)",
                    min_value=-5.0,
                    max_value=0.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Rayleigh-Bénard convection pattern formation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        r = parameters.get("r", 0.1)
        k = parameters.get("k", 1.0)
        g2 = parameters.get("g2", 0.0)
        g3 = parameters.get("g3", -1.0)

        k_sq = k * k

        # Swift-Hohenberg: du/dt = r*u - (k² + ∇²)²u + g₂u² + g₃u³
        # Expanding (k² + ∇²)²u = k⁴u + 2k²∇²u + ∇⁴u
        rhs = (
            f"{r} * u - {k_sq**2} * u - 2 * {k_sq} * laplace(u) - laplace(laplace(u))"
        )
        if g2 != 0:
            rhs += f" + {g2} * u**2"
        if g3 != 0:
            rhs += f" + {g3} * u**3"

        return PDE(
            rhs={"u": rhs},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )


@register_pde("burgers")
class BurgersPDE(ScalarPDEPreset):
    """Burgers' equation.

    The viscous Burgers' equation describes shock wave formation:

        du/dt = nu * laplace(u) - u * d_dx(u)

    where nu is viscosity.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="burgers",
            category="physics",
            description="Burgers' equation (viscous)",
            equations={
                "u": "nu * laplace(u) - u * d_dx(u)",
            },
            parameters=[
                PDEParameter(
                    name="nu",
                    default=0.01,
                    description="Viscosity",
                    min_value=0.001,
                    max_value=1.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Shock wave formation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        nu = parameters.get("nu", 0.01)

        return PDE(
            rhs={"u": f"{nu} * laplace(u) - u * d_dx(u)"},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

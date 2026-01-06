"""Swift-Hohenberg equation for pattern formation."""

from typing import Any

from pde import PDE, CartesianGrid

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("swift-hohenberg")
class SwiftHohenbergPDE(ScalarPDEPreset):
    """Swift-Hohenberg equation for pattern formation.

    The Swift-Hohenberg equation is a model for pattern formation:

        du/dt = r*u - (k²+laplace)²*u + g₂*u² + g₃*u³ + g₅*u⁵

    where:
        - r is the control parameter
        - k is the critical wavenumber
        - g₂, g₃, g₅ are nonlinear coefficients
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="swift-hohenberg",
            category="physics",
            description="Swift-Hohenberg pattern formation",
            equations={
                "u": "r*u - (k**2 + laplace(u))**2 + g2*u**2 + g3*u**3 + g5*u**5",
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
                    description="Cubic nonlinearity (should be negative for stability if g5=0)",
                    min_value=-5.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="g5",
                    default=0.0,
                    description="Quintic nonlinearity (should be negative for stability)",
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
        g5 = parameters.get("g5", 0.0)

        k_sq = k * k

        # Swift-Hohenberg: du/dt = r*u - (k² + ∇²)²u + g₂u² + g₃u³ + g₅u⁵
        # Expanding (k² + ∇²)²u = k⁴u + 2k²∇²u + ∇⁴u
        rhs = (
            f"{r} * u - {k_sq**2} * u - 2 * {k_sq} * laplace(u) - laplace(laplace(u))"
        )
        if g2 != 0:
            rhs += f" + {g2} * u**2"
        if g3 != 0:
            rhs += f" + {g3} * u**3"
        if g5 != 0:
            rhs += f" + {g5} * u**5"

        return PDE(
            rhs={"u": rhs},
            bc=self._convert_bc(bc),
        )

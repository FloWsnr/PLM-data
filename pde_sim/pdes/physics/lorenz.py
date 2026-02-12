"""Diffusively coupled Lorenz system."""

from typing import Any

from pde import PDE, CartesianGrid

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("lorenz")
class LorenzPDE(MultiFieldPDEPreset):
    """Diffusively coupled Lorenz system.

    Based on visualpde.com formulation:

        dX/dt = D * laplace(X) + sigma * (Y - X)
        dY/dt = D * laplace(Y) + X * (rho - Z) - Y
        dZ/dt = D * laplace(Z) + X * Y - beta * Z

    A spatially extended version of the famous Lorenz equations,
    exhibiting spatiotemporal chaos through the interplay of local
    chaotic dynamics and diffusive coupling.

    Behaviors based on coupling strength D:
        - Weak coupling (D < 0.2): fragmented chaos with localized patches
        - Intermediate (D ~ 0.5): complex spatiotemporal patterns
        - Strong coupling (D > 2): synchronized oscillations

    Physical interpretation (atmospheric convection):
        - X: convective circulation intensity
        - Y: temperature difference (ascending vs descending)
        - Z: vertical temperature profile deviation

    Reference: Lorenz (1963), Cross & Hohenberg (1993)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="lorenz",
            category="physics",
            description="Diffusively coupled Lorenz system (spatiotemporal chaos)",
            equations={
                "X": "D * laplace(X) + sigma * (Y - X)",
                "Y": "D * laplace(Y) + X * (rho - Z) - Y",
                "Z": "D * laplace(Z) + X * Y - beta * Z",
            },
            parameters=[
                PDEParameter("sigma", "Prandtl number"),
                PDEParameter("rho", "Rayleigh number (normalized)"),
                PDEParameter("beta", "Geometric factor"),
                PDEParameter("D", "Diffusion coefficient (coupling strength)"),
            ],
            num_fields=3,
            field_names=["X", "Y", "Z"],
            reference="Lorenz (1963) Deterministic nonperiodic flow",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Lorenz PDE system.

        Args:
            parameters: Dictionary with sigma, rho, beta, D.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        sigma = parameters.get("sigma", 10.0)
        rho = parameters.get("rho", 30.0)
        beta = parameters.get("beta", 8.0 / 3.0)
        D = parameters.get("D", 0.5)

        return PDE(
            rhs={
                "X": f"{D} * laplace(X) + {sigma} * (Y - X)",
                "Y": f"{D} * laplace(Y) + X * ({rho} - Z) - Y",
                "Z": f"{D} * laplace(Z) + X * Y - {beta} * Z",
            },
            bc=self._convert_bc(bc),
        )

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        sigma = parameters.get("sigma", 10.0)
        rho = parameters.get("rho", 30.0)
        beta = parameters.get("beta", 8.0 / 3.0)
        D = parameters.get("D", 0.5)

        return {
            "X": f"{D} * laplace(X) + {sigma} * (Y - X)",
            "Y": f"{D} * laplace(Y) + X * ({rho} - Z) - Y",
            "Z": f"{D} * laplace(Z) + X * Y - {beta} * Z",
        }

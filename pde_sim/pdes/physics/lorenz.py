"""Diffusively coupled Lorenz system."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

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
                PDEParameter(
                    name="sigma",
                    default=10.0,
                    description="Prandtl number",
                    min_value=1.0,
                    max_value=20.0,
                ),
                PDEParameter(
                    name="rho",
                    default=30.0,
                    description="Rayleigh number (normalized)",
                    min_value=1.0,
                    max_value=50.0,
                ),
                PDEParameter(
                    name="beta",
                    default=8.0 / 3.0,
                    description="Geometric factor",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="D",
                    default=0.5,
                    description="Diffusion coefficient (coupling strength)",
                    min_value=0.0,
                    max_value=5.0,
                ),
            ],
            num_fields=3,
            field_names=["X", "Y", "Z"],
            reference="Lorenz (1963) Deterministic nonperiodic flow",
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

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial state near Lorenz attractor.

        Default: X = 0.3*RANDN + 1, Y = 0, Z = 29.
        """
        seed = ic_params.get("seed")
        if seed is not None:
            np.random.seed(seed)
        noise = ic_params.get("noise", 0.3)

        # Reference uses X = 0.3*RANDN + 1, Y = 0, Z = 29
        x_data = noise * np.random.randn(*grid.shape) + 1.0
        y_data = np.zeros(grid.shape)
        z_data = np.full(grid.shape, 29.0)

        X_field = ScalarField(grid, x_data)
        X_field.label = "X"
        Y_field = ScalarField(grid, y_data)
        Y_field.label = "Y"
        Z_field = ScalarField(grid, z_data)
        Z_field.label = "Z"

        return FieldCollection([X_field, Y_field, Z_field])

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

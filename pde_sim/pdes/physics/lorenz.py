"""Diffusively coupled Lorenz system."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("lorenz")
class LorenzPDE(MultiFieldPDEPreset):
    """Diffusively coupled Lorenz system.

    The Lorenz equations with spatial diffusion coupling:

        dX/dt = Dx * laplace(X) + sigma * (Y - X)
        dY/dt = Dy * laplace(Y) + X * (rho - Z) - Y
        dZ/dt = Dz * laplace(Z) + X * Y - beta * Z

    where:
        - X, Y, Z are the Lorenz variables (capitalized to avoid collision with grid coordinates)
        - sigma, rho, beta are the classic Lorenz parameters
        - Dx, Dy, Dz are diffusion coefficients

    Creates spatiotemporal chaos from coupled chaotic oscillators.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="lorenz",
            category="physics",
            description="Diffusively coupled Lorenz system",
            equations={
                "X": "Dx * laplace(X) + sigma * (Y - X)",
                "Y": "Dy * laplace(Y) + X * (rho - Z) - Y",
                "Z": "Dz * laplace(Z) + X * Y - beta * Z",
            },
            parameters=[
                PDEParameter(
                    name="sigma",
                    default=10.0,
                    description="Lorenz sigma parameter",
                    min_value=1.0,
                    max_value=20.0,
                ),
                PDEParameter(
                    name="rho",
                    default=28.0,
                    description="Lorenz rho parameter",
                    min_value=1.0,
                    max_value=50.0,
                ),
                PDEParameter(
                    name="beta",
                    default=8.0 / 3.0,
                    description="Lorenz beta parameter",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dx",
                    default=0.1,
                    description="Diffusion of x",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dy",
                    default=0.1,
                    description="Diffusion of y",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dz",
                    default=0.1,
                    description="Diffusion of z",
                    min_value=0.0,
                    max_value=10.0,
                ),
            ],
            num_fields=3,
            field_names=["X", "Y", "Z"],
            reference="Lorenz (1963) attractor with diffusion",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        sigma = parameters.get("sigma", 10.0)
        rho = parameters.get("rho", 28.0)
        beta = parameters.get("beta", 8.0 / 3.0)
        Dx = parameters.get("Dx", 0.1)
        Dy = parameters.get("Dy", 0.1)
        Dz = parameters.get("Dz", 0.1)

        return PDE(
            rhs={
                "X": f"{Dx} * laplace(X) + {sigma} * (Y - X)",
                "Y": f"{Dy} * laplace(Y) + X * ({rho} - Z) - Y",
                "Z": f"{Dz} * laplace(Z) + X * Y - {beta} * Z",
            },
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state near Lorenz attractor."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.5)

        # Start near one fixed point with perturbations
        x_data = 1.0 + noise * np.random.randn(*grid.shape)
        y_data = 1.0 + noise * np.random.randn(*grid.shape)
        z_data = 1.0 + noise * np.random.randn(*grid.shape)

        X_field = ScalarField(grid, x_data)
        X_field.label = "X"
        Y_field = ScalarField(grid, y_data)
        Y_field.label = "Y"
        Z_field = ScalarField(grid, z_data)
        Z_field.label = "Z"

        return FieldCollection([X_field, Y_field, Z_field])

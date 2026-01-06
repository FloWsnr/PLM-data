"""Perona-Malik anisotropic diffusion."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("perona-malik")
class PeronaMalikPDE(ScalarPDEPreset):
    """Perona-Malik anisotropic diffusion with exponential diffusivity.

    Based on visualpde.com formulation:

        du/dt = div(exp(-D * |grad(u)|²) * grad(u))

    The exponential diffusivity function strongly suppresses diffusion
    at edges (where |grad(u)|² is large) while allowing full diffusion
    in smooth regions.

    Approximated as:
        du/dt = laplace(u) * exp(-D * gradient_squared(u))

    This captures the key feature of edge-preserving smoothing where
    diffusion is exponentially reduced at sharp gradients.

    Reference: https://visualpde.com/nonlinear-physics/perona-malik
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="perona-malik",
            category="physics",
            description="Perona-Malik edge-preserving diffusion (exponential)",
            equations={"u": "laplace(u) * exp(-D * gradient_squared(u))"},
            parameters=[
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Edge sensitivity (higher = more edge preservation)",
                    min_value=0.1,
                    max_value=10.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="https://visualpde.com/nonlinear-physics/perona-malik",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)

        # Perona-Malik with exponential diffusivity:
        # du/dt = laplace(u) * exp(-D * |grad(u)|²)
        return PDE(
            rhs={"u": f"laplace(u) * exp(-{D} * gradient_squared(u))"},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial image-like pattern with edges."""
        if ic_type in ("perona-malik-default", "default"):
            np.random.seed(ic_params.get("seed"))

            x, y = np.meshgrid(
                np.linspace(0, 1, grid.shape[0]),
                np.linspace(0, 1, grid.shape[1]),
                indexing="ij",
            )

            # Step pattern with noise (like a noisy image with edges)
            data = np.zeros(grid.shape)
            data[x > 0.3] = 0.5
            data[x > 0.7] = 1.0
            data[y > 0.5] += 0.3

            # Add noise
            data += 0.1 * np.random.randn(*grid.shape)
            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

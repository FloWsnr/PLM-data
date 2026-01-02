"""Perona-Malik anisotropic diffusion."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("perona-malik")
class PeronaMalikPDE(ScalarPDEPreset):
    """Perona-Malik anisotropic diffusion.

    Image processing PDE for edge-preserving smoothing:

        du/dt = div(g(|grad(u)|) * grad(u))

    where g(s) = 1/(1 + (s/K)^2) is the diffusivity function.

    Simplified version:
        du/dt = D * laplace(u) - D * K * gradient_squared(u) / (K^2 + gradient_squared(u))
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="perona-malik",
            category="physics",
            description="Perona-Malik edge-preserving diffusion",
            equations={"u": "D * laplace(u) / (1 + gradient_squared(u) / K^2)"},
            parameters=[
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Diffusion coefficient",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="K",
                    default=1.0,
                    description="Edge threshold",
                    min_value=0.1,
                    max_value=10.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Perona & Malik (1990) anisotropic diffusion",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        K = parameters.get("K", 1.0)
        K_sq = K**2

        # Simplified: linear diffusion with edge-dependent reduction
        return PDE(
            rhs={"u": f"{D} * laplace(u) / (1 + gradient_squared(u) / {K_sq})"},
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

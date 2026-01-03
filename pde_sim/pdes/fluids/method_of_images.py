"""Potential flow near a wall using method of images concept."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("method-of-images")
class MethodOfImagesPDE(ScalarPDEPreset):
    """Potential flow near a wall using method of images concept.

    Vortex dynamics with wall reflection symmetry:

        dw/dt = nu * laplace(w)

    Initial condition uses mirror vortices to simulate wall effect.
    The dynamics evolve the vorticity with viscous diffusion.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="method-of-images",
            category="fluids",
            description="Vortex near wall (method of images)",
            equations={"w": "nu * laplace(w)"},
            parameters=[
                PDEParameter(
                    name="nu",
                    default=0.01,
                    description="Kinematic viscosity",
                    min_value=0.001,
                    max_value=0.1,
                ),
                PDEParameter(
                    name="wall_distance",
                    default=0.2,
                    description="Distance of vortex from wall",
                    min_value=0.05,
                    max_value=0.4,
                ),
            ],
            num_fields=1,
            field_names=["w"],
            reference="Potential flow method of images",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        nu = parameters.get("nu", 0.01)

        return PDE(
            rhs={"w": f"{nu} * laplace(w)"},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create vortex pair near wall (real + image)."""
        wall_dist = ic_params.get("wall_distance", 0.2)
        strength = ic_params.get("strength", 10.0)
        radius = ic_params.get("radius", 0.08)

        x, y = np.meshgrid(
            np.linspace(0, 1, grid.shape[0]),
            np.linspace(0, 1, grid.shape[1]),
            indexing="ij",
        )

        # Wall at y = 0, vortex at y = wall_dist
        # Image vortex at y = -wall_dist (reflected, opposite sign)
        y_vortex = wall_dist
        y_image = -wall_dist  # Outside domain, but affects field

        # Real vortex
        r1_sq = (x - 0.5) ** 2 + (y - y_vortex) ** 2
        w1 = strength * np.exp(-r1_sq / (2 * radius**2))

        # Image vortex (opposite circulation)
        r2_sq = (x - 0.5) ** 2 + (y - y_image) ** 2
        w2 = -strength * np.exp(-r2_sq / (2 * radius**2))

        # Add mirror on the other side for periodic BC
        r3_sq = (x - 0.5) ** 2 + (y - (1.0 - wall_dist)) ** 2
        w3 = strength * np.exp(-r3_sq / (2 * radius**2))
        r4_sq = (x - 0.5) ** 2 + (y - (1.0 + wall_dist)) ** 2
        w4 = -strength * np.exp(-r4_sq / (2 * radius**2))

        data = w1 + w2 + w3 + w4
        return ScalarField(grid, data)

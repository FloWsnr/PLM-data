"""Vortex dipole dynamics."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("dipoles")
class DipolesPDE(ScalarPDEPreset):
    """Vortex dipole dynamics.

    Two co-rotating vortices that move together as a dipole:

        dw/dt = nu * laplace(w)

    The dipole structure creates self-induced motion.
    With diffusion, the dipoles gradually decay.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="dipoles",
            category="fluids",
            description="Vortex dipole motion and decay",
            equations={"w": "nu * laplace(w)"},
            parameters=[
                PDEParameter(
                    name="nu",
                    default=0.005,
                    description="Kinematic viscosity",
                    min_value=0.001,
                    max_value=0.1,
                ),
                PDEParameter(
                    name="separation",
                    default=0.15,
                    description="Vortex separation distance",
                    min_value=0.05,
                    max_value=0.3,
                ),
            ],
            num_fields=1,
            field_names=["w"],
            reference="Vortex dipole dynamics",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        nu = parameters.get("nu", 0.005)

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
        """Create vortex dipole (two opposite-sign vortices)."""
        separation = ic_params.get("separation", 0.15)
        strength = ic_params.get("strength", 10.0)
        radius = ic_params.get("radius", 0.05)

        x, y = np.meshgrid(
            np.linspace(0, 1, grid.shape[0]),
            np.linspace(0, 1, grid.shape[1]),
            indexing="ij",
        )

        # Dipole centered at (0.3, 0.5), moving to the right
        cx, cy = 0.3, 0.5

        # Two vortices separated vertically
        y1 = cy + separation / 2
        y2 = cy - separation / 2

        r1_sq = (x - cx) ** 2 + (y - y1) ** 2
        r2_sq = (x - cx) ** 2 + (y - y2) ** 2

        # Opposite circulations create rightward motion
        w1 = strength * np.exp(-r1_sq / (2 * radius**2))
        w2 = -strength * np.exp(-r2_sq / (2 * radius**2))

        data = w1 + w2
        return ScalarField(grid, data)

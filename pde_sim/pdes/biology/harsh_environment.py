"""Population in harsh environment with Allee effect."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("harsh-environment")
class HarshEnvironmentPDE(ScalarPDEPreset):
    """Population in harsh environment with Allee effect.

    Population dynamics with strong Allee effect:

        du/dt = D * laplace(u) + r * u * (u - theta) * (1 - u)

    where:
        - u is population density (normalized)
        - D is diffusion coefficient
        - r is growth rate
        - theta is Allee threshold (below which population declines)

    Exhibits bistability: extinction or survival depending on initial density.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="harsh-environment",
            category="biology",
            description="Allee effect in harsh environment",
            equations={"u": "D * laplace(u) + r * u * (u - theta) * (1 - u)"},
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.1,
                    description="Diffusion coefficient",
                    min_value=0.01,
                    max_value=0.5,
                ),
                PDEParameter(
                    name="r",
                    default=1.0,
                    description="Growth rate",
                    min_value=0.1,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="theta",
                    default=0.2,
                    description="Allee threshold",
                    min_value=0.0,
                    max_value=0.5,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Strong Allee effect model",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 0.1)
        r = parameters.get("r", 1.0)
        theta = parameters.get("theta", 0.2)

        return PDE(
            rhs={"u": f"{D} * laplace(u) + {r} * u * (u - {theta}) * (1 - u)"},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial population patch."""
        if ic_type in ("harsh-environment-default", "default"):
            # Population patch above Allee threshold
            x, y = np.meshgrid(
                np.linspace(0, 1, grid.shape[0]),
                np.linspace(0, 1, grid.shape[1]),
                indexing="ij",
            )
            r_sq = (x - 0.5) ** 2 + (y - 0.5) ** 2
            data = 0.8 * np.exp(-r_sq / 0.05)
            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

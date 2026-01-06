"""Population in harsh environment - logistic growth with boundary effects."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("harsh-environment")
class HarshEnvironmentPDE(ScalarPDEPreset):
    """Population in harsh environment - logistic reaction-diffusion.

    Standard logistic growth from visualpde.com:

        du/dt = D*∇²u + u*(1 - u/K)

    The "harsh environment" comes from BOUNDARY CONDITIONS, not the
    reaction term. With Dirichlet BCs (u=0 at boundaries), populations
    must overcome both diffusion loss and boundary death.

    Key result: Population persists if and only if D < L²/(2π²), where L
    is the domain size. For larger D, boundary effects dominate and the
    population goes extinct.

    Reference: visualpde.com harsh environment
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="harsh-environment",
            category="biology",
            description="Logistic growth with harsh boundary conditions",
            equations={"u": "D * laplace(u) + u * (1 - u/K)"},
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
                    description="Carrying capacity",
                    min_value=0.1,
                    max_value=10.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="visualpde.com harsh environment",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        K = parameters.get("K", 1.0)

        return PDE(
            rhs={"u": f"{D} * laplace(u) + u * (1 - u / {K})"},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial population - localized patches."""
        if ic_type in ("harsh-environment-default", "default"):
            # Multiple population patches
            np.random.seed(ic_params.get("seed"))
            x, y = np.meshgrid(
                np.linspace(0, 1, grid.shape[0]),
                np.linspace(0, 1, grid.shape[1]),
                indexing="ij",
            )
            # Create a few patches
            data = np.zeros(grid.shape)
            for cx, cy in [(0.3, 0.3), (0.7, 0.5), (0.5, 0.7)]:
                r_sq = (x - cx) ** 2 + (y - cy) ** 2
                data += 0.8 * np.exp(-r_sq / 0.02)
            return ScalarField(grid, np.clip(data, 0, 1))

        return create_initial_condition(grid, ic_type, ic_params)

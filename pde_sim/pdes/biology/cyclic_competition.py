"""Cyclic competition (rock-paper-scissors) model."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("cyclic-competition")
class CyclicCompetitionPDE(MultiFieldPDEPreset):
    """Cyclic competition (rock-paper-scissors) model.

    Three-species competition where each species dominates one other:
    A beats B, B beats C, C beats A (like rock-paper-scissors).

        du/dt = Du * laplace(u) + u * (1 - u - v - w) - alpha * u * v
        dv/dt = Dv * laplace(v) + v * (1 - u - v - w) - alpha * v * w
        dw/dt = Dw * laplace(w) + w * (1 - u - v - w) - alpha * w * u

    where:
        - u, v, w are species densities
        - D is diffusion coefficient
        - alpha is competition strength

    Exhibits spiral wave patterns and biodiversity maintenance.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="cyclic-competition",
            category="biology",
            description="Rock-paper-scissors cyclic competition",
            equations={
                "u": "D * laplace(u) + u * (1 - u - v - w) - alpha * u * v",
                "v": "D * laplace(v) + v * (1 - u - v - w) - alpha * v * w",
                "w": "D * laplace(w) + w * (1 - u - v - w) - alpha * w * u",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.1,
                    description="Diffusion coefficient",
                    min_value=0.001,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="alpha",
                    default=1.0,
                    description="Competition strength",
                    min_value=0.1,
                    max_value=10.0,
                ),
            ],
            num_fields=3,
            field_names=["u", "v", "w"],
            reference="May-Leonard cyclic competition model",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 0.1)
        alpha = parameters.get("alpha", 1.0)

        return PDE(
            rhs={
                "u": f"{D} * laplace(u) + u * (1 - u - v - w) - {alpha} * u * v",
                "v": f"{D} * laplace(v) + v * (1 - u - v - w) - {alpha} * v * w",
                "w": f"{D} * laplace(w) + w * (1 - u - v - w) - {alpha} * w * u",
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state with three species in different regions."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.1)

        # Initialize with random small values
        u_data = 0.33 + noise * np.random.randn(*grid.shape)
        v_data = 0.33 + noise * np.random.randn(*grid.shape)
        w_data = 0.33 + noise * np.random.randn(*grid.shape)

        # Ensure non-negative
        u_data = np.clip(u_data, 0.01, 1.0)
        v_data = np.clip(v_data, 0.01, 1.0)
        w_data = np.clip(w_data, 0.01, 1.0)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"
        w = ScalarField(grid, w_data)
        w.label = "w"

        return FieldCollection([u, v, w])

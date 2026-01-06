"""Cyclic competition (rock-paper-scissors) model."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("cyclic-competition")
class CyclicCompetitionPDE(MultiFieldPDEPreset):
    """Cyclic competition (rock-paper-scissors) model.

    Generalized Lotka-Volterra system from visualpde.com with asymmetric
    competition creating cyclic dominance (rock-paper-scissors dynamics):

        du/dt = Du*∇²u + u*(1 - u - a*v - b*w)
        dv/dt = Dv*∇²v + v*(1 - b*u - v - a*w)
        dw/dt = Dw*∇²w + w*(1 - a*u - b*v - w)

    where a < 1 < b creates the cyclic dominance:
        - u beats v (coeff a < 1 means v is weak competitor)
        - v beats w
        - w beats u

    Exhibits spiral wave patterns and biodiversity maintenance.

    Reference: visualpde.com cyclic competition
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="cyclic-competition",
            category="biology",
            description="Rock-paper-scissors cyclic competition",
            equations={
                "u": "Du * laplace(u) + u * (1 - u - a*v - b*w)",
                "v": "Dv * laplace(v) + v * (1 - b*u - v - a*w)",
                "w": "Dw * laplace(w) + w * (1 - a*u - b*v - w)",
            },
            parameters=[
                PDEParameter(
                    name="Du",
                    default=0.5,
                    description="Diffusion coefficient for u",
                    min_value=0.01,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="Dv",
                    default=0.5,
                    description="Diffusion coefficient for v",
                    min_value=0.01,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="Dw",
                    default=0.5,
                    description="Diffusion coefficient for w",
                    min_value=0.01,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="a",
                    default=0.5,
                    description="Weak competition coefficient (a < 1)",
                    min_value=0.1,
                    max_value=0.99,
                ),
                PDEParameter(
                    name="b",
                    default=1.5,
                    description="Strong competition coefficient (b > 1)",
                    min_value=1.01,
                    max_value=3.0,
                ),
            ],
            num_fields=3,
            field_names=["u", "v", "w"],
            reference="visualpde.com cyclic competition",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Du = parameters.get("Du", 0.5)
        Dv = parameters.get("Dv", 0.5)
        Dw = parameters.get("Dw", 0.5)
        a = parameters.get("a", 0.5)
        b = parameters.get("b", 1.5)

        return PDE(
            rhs={
                "u": f"{Du} * laplace(u) + u * (1 - u - {a}*v - {b}*w)",
                "v": f"{Dv} * laplace(v) + v * (1 - {b}*u - v - {a}*w)",
                "w": f"{Dw} * laplace(w) + w * (1 - {a}*u - {b}*v - w)",
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

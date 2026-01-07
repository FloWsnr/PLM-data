"""Cyclic competition (rock-paper-scissors) model with spiral waves."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("cyclic-competition")
class CyclicCompetitionPDE(MultiFieldPDEPreset):
    """Cyclic competition (rock-paper-scissors) model.

    Three-species Lotka-Volterra with cyclic dominance:

        du/dt = Du*laplace(u) + u*(1 - u - a*v - b*w)
        dv/dt = Dv*laplace(v) + v*(1 - b*u - v - a*w)
        dw/dt = Dw*laplace(w) + w*(1 - a*u - b*v - w)

    where a < 1 < b creates cyclic dominance:
        - u beats v (coefficient a < 1 means v is weak)
        - v beats w
        - w beats u

    Key phenomena:
        - Spiral waves: dominant feature from structured ICs
        - Biodiversity maintenance: all three species coexist
        - Critical mobility threshold: high diffusion can collapse diversity

    References:
        May & Leonard (1975). SIAM J. Appl. Math., 29(2), 243-253.
        Reichenbach, Mobilia & Frey (2007). Nature, 448(7157), 1046-1049.
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
                    name="a",
                    default=0.8,
                    description="Weak competition coefficient (a < 1)",
                    min_value=0.1,
                    max_value=0.99,
                ),
                PDEParameter(
                    name="b",
                    default=1.9,
                    description="Strong competition coefficient (b > 1)",
                    min_value=1.01,
                    max_value=3.0,
                ),
                PDEParameter(
                    name="Du",
                    default=2.0,
                    description="Diffusion coefficient for u",
                    min_value=0.01,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="Dv",
                    default=0.5,
                    description="Diffusion coefficient for v",
                    min_value=0.01,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="Dw",
                    default=0.5,
                    description="Diffusion coefficient for w",
                    min_value=0.01,
                    max_value=5.0,
                ),
            ],
            num_fields=3,
            field_names=["u", "v", "w"],
            reference="May & Leonard (1975), Reichenbach et al. (2007)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        a = parameters.get("a", 0.8)
        b = parameters.get("b", 1.9)
        Du = parameters.get("Du", 2.0)
        Dv = parameters.get("Dv", 0.5)
        Dw = parameters.get("Dw", 0.5)

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
        **kwargs,
    ) -> FieldCollection:
        """Create initial state - sech bump of all species at center.

        Uses sech(r) = 1/cosh(r) profile matching Visual PDE reference.
        """
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.01)

        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        cx = (x_bounds[0] + x_bounds[1]) / 2
        cy = (y_bounds[0] + y_bounds[1]) / 2

        # sech(r) profile: 1/cosh(sqrt((x-cx)^2 + (y-cy)^2))
        # This matches Visual PDE: 1/cosh(sqrt((x-L_x/2)^2+(y-L_y/2)^2))
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        bump = 1.0 / np.cosh(r)

        u_data = bump + noise * np.random.randn(*grid.shape)
        v_data = bump + noise * np.random.randn(*grid.shape)
        w_data = bump + noise * np.random.randn(*grid.shape)

        # Ensure non-negative
        u_data = np.clip(u_data, 0.0, None)
        v_data = np.clip(v_data, 0.0, None)
        w_data = np.clip(w_data, 0.0, None)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"
        w = ScalarField(grid, w_data)
        w.label = "w"

        return FieldCollection([u, v, w])

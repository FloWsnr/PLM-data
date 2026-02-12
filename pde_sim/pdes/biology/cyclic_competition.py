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
                PDEParameter("a", "Weak competition coefficient (a < 1)"),
                PDEParameter("b", "Strong competition coefficient (b > 1)"),
                PDEParameter("Du", "Diffusion coefficient for u"),
                PDEParameter("Dv", "Diffusion coefficient for v"),
                PDEParameter("Dw", "Diffusion coefficient for w"),
            ],
            num_fields=3,
            field_names=["u", "v", "w"],
            reference="May & Leonard (1975), Reichenbach et al. (2007)",
            supported_dimensions=[1, 2, 3],
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
        if ic_type not in ("default",):
            return super().create_initial_state(grid, ic_type, ic_params, **kwargs)

        rng = np.random.default_rng(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.01)

        ndim = len(grid.shape)
        coords_1d = [np.linspace(grid.axes_bounds[i][0], grid.axes_bounds[i][1], grid.shape[i]) for i in range(ndim)]
        coords = np.meshgrid(*coords_1d, indexing="ij")

        center = [(grid.axes_bounds[i][0] + grid.axes_bounds[i][1]) / 2 for i in range(ndim)]
        r = np.sqrt(sum((c - ctr)**2 for c, ctr in zip(coords, center)))
        bump = 1.0 / np.cosh(r)

        u_data = bump + noise * rng.standard_normal(grid.shape)
        v_data = bump + noise * rng.standard_normal(grid.shape)
        w_data = bump + noise * rng.standard_normal(grid.shape)

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

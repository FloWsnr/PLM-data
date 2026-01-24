"""Cyclic competition wave variant - equal diffusion with edge-localized initial conditions."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("cyclic-competition-wave")
class CyclicCompetitionWavePDE(MultiFieldPDEPreset):
    """Cyclic competition wave variant.

    Demonstrates wave-induced spatiotemporal chaos without Turing instability.
    Uses equal diffusion for all species (no Turing pattern formation) but
    creates traveling waves from localized initial conditions.

    Equations are the same as cyclic-competition:

        du/dt = D*laplace(u) + u*(1 - u - a*v - b*w)
        dv/dt = D*laplace(v) + v*(1 - b*u - v - a*w)
        dw/dt = D*laplace(w) + w*(1 - a*u - b*v - w)

    Key difference from cyclic-competition:
        - Equal diffusion (Du = Dv = Dw = D) prevents Turing patterns
        - Edge-localized initial conditions create expanding waves
        - Spatiotemporal chaos emerges from wave interactions

    Initial condition: Species present only in left 10% of domain,
    simulating an invasion front.

    Reference:
        https://visualpde.com/nonlinear-dynamics/cyclic-competition-wave
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="cyclic-competition-wave",
            category="biology",
            description="Cyclic competition with equal diffusion (wave chaos)",
            equations={
                "u": "D * laplace(u) + u * (1 - u - a*v - b*w)",
                "v": "D * laplace(v) + v * (1 - b*u - v - a*w)",
                "w": "D * laplace(w) + w * (1 - a*u - b*v - w)",
            },
            parameters=[
                PDEParameter("a", "Weak competition coefficient (a < 1)"),
                PDEParameter("b", "Strong competition coefficient (b > 1)"),
                PDEParameter("D", "Equal diffusion coefficient for all species"),
            ],
            num_fields=3,
            field_names=["u", "v", "w"],
            reference="VisualPDE cyclicCompetitionWave",
            supported_dimensions=[2],  # Currently 2D only (IC uses 2D meshgrid)
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        a = parameters.get("a", 0.8)
        b = parameters.get("b", 1.9)
        D = parameters.get("D", 0.3)

        return PDE(
            rhs={
                "u": f"{D} * laplace(u) + u * (1 - u - {a}*v - {b}*w)",
                "v": f"{D} * laplace(v) + v * (1 - {b}*u - v - {a}*w)",
                "w": f"{D} * laplace(w) + w * (1 - {a}*u - {b}*v - w)",
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
        """Create edge-localized initial state (Heaviside-type).

        Species present only in left 10% of domain, simulating invasion.
        Matches Visual PDE: u = H(0.1-x/L_x)*(1+0.001*RANDN)
        """
        # For standard IC types, use parent class implementation
        if ic_type not in ("default",):
            return super().create_initial_state(grid, ic_type, ic_params, **kwargs)

        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.001)
        edge_fraction = ic_params.get("edge_fraction", 0.1)

        x_bounds = grid.axes_bounds[0]
        L_x = x_bounds[1] - x_bounds[0]
        threshold_x = x_bounds[0] + edge_fraction * L_x

        # Create coordinate arrays
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y_bounds = grid.axes_bounds[1]
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Heaviside mask: 1 for x < threshold, 0 otherwise
        mask = (X < threshold_x).astype(float)

        # u: H(0.1-x/L_x)*(1+noise*RANDN)
        u_data = mask * (1 + noise * np.random.randn(*grid.shape))

        # v: H(0.1-x/L_x) - no noise in Visual PDE reference
        v_data = mask.copy()

        # w: H(0.1-x/L_x) - no noise in Visual PDE reference
        w_data = mask.copy()

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

"""Shallow water equation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import PDEMetadata, PDEParameter, ScalarPDEPreset
from .. import register_pde


@register_pde("shallow-water")
class ShallowWaterPDE(ScalarPDEPreset):
    """Linearized Shallow Water equation (height component).

    The shallow water equations describe free surface flows:

        dh/dt = -H * (d_dx(u) + d_dy(v))
        du/dt = -g * d_dx(h)
        dv/dt = -g * d_dy(h)

    This simplified version models the height evolution with
    a diffusion-like approximation:

        dh/dt = c^2 * laplace(h)

    where c = sqrt(g*H) is the wave speed.

    For full shallow water dynamics, see the multi-field version.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="shallow-water",
            category="fluids",
            description="Linearized shallow water wave equation",
            equations={"h": "c**2 * laplace(h)"},
            parameters=[
                PDEParameter(
                    name="c",
                    default=1.0,
                    description="Wave speed sqrt(g*H)",
                    min_value=0.1,
                    max_value=10.0,
                ),
            ],
            num_fields=1,
            field_names=["h"],
            reference="Shallow water wave propagation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        c = parameters.get("c", 1.0)
        c_sq = c**2

        bc_spec = "periodic" if bc.get("x") == "periodic" else "no-flux"

        return PDE(
            rhs={"h": f"{c_sq} * laplace(h)"},
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial water height perturbation."""
        if ic_type in ("shallow-water-default", "drop"):
            # Water drop (Gaussian perturbation)
            x0 = ic_params.get("x0", 0.5)
            y0 = ic_params.get("y0", 0.5)
            amplitude = ic_params.get("amplitude", 0.5)
            width = ic_params.get("width", 0.1)
            background = ic_params.get("background", 1.0)

            x, y = np.meshgrid(
                np.linspace(0, 1, grid.shape[0]),
                np.linspace(0, 1, grid.shape[1]),
                indexing="ij",
            )

            r_sq = (x - x0) ** 2 + (y - y0) ** 2
            data = background + amplitude * np.exp(-r_sq / (2 * width**2))
            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

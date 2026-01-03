"""Vorticity equation for 2D incompressible flow."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import PDEMetadata, PDEParameter, ScalarPDEPreset
from .. import register_pde


@register_pde("vorticity")
class VorticityPDE(ScalarPDEPreset):
    """2D Vorticity equation (simplified Navier-Stokes).

    The vorticity formulation of 2D incompressible Navier-Stokes:

        dw/dt = nu * laplace(w) - u * d_dx(w) - v * d_dy(w)

    where:
        - w is the vorticity (curl of velocity)
        - nu is the kinematic viscosity
        - (u, v) is the velocity field

    In 2D, velocity can be derived from stream function psi:
        w = laplace(psi), u = d_dy(psi), v = -d_dx(psi)

    This simplified version uses pure diffusion of vorticity,
    suitable for decay problems and demonstrating viscous effects:

        dw/dt = nu * laplace(w)

    For full advection, the stream function must be solved at each step.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="vorticity",
            category="fluids",
            description="2D vorticity diffusion equation",
            equations={"w": "nu * laplace(w)"},
            parameters=[
                PDEParameter(
                    name="nu",
                    default=0.01,
                    description="Kinematic viscosity",
                    min_value=0.001,
                    max_value=0.1,
                ),
            ],
            num_fields=1,
            field_names=["w"],
            reference="2D Navier-Stokes vorticity formulation",
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
        """Create initial vorticity distribution."""
        if ic_type in ("vorticity-default", "vortex-pair"):
            # Two counter-rotating vortices
            x0_1, y0_1 = ic_params.get("x1", 0.35), ic_params.get("y1", 0.5)
            x0_2, y0_2 = ic_params.get("x2", 0.65), ic_params.get("y2", 0.5)
            strength = ic_params.get("strength", 10.0)
            radius = ic_params.get("radius", 0.1)

            x, y = np.meshgrid(
                np.linspace(0, 1, grid.shape[0]),
                np.linspace(0, 1, grid.shape[1]),
                indexing="ij",
            )

            # Gaussian vortices
            r1_sq = (x - x0_1) ** 2 + (y - y0_1) ** 2
            r2_sq = (x - x0_2) ** 2 + (y - y0_2) ** 2

            w1 = strength * np.exp(-r1_sq / (2 * radius**2))
            w2 = -strength * np.exp(-r2_sq / (2 * radius**2))

            data = w1 + w2
            return ScalarField(grid, data)

        elif ic_type == "single-vortex":
            x0 = ic_params.get("x0", 0.5)
            y0 = ic_params.get("y0", 0.5)
            strength = ic_params.get("strength", 10.0)
            radius = ic_params.get("radius", 0.15)

            x, y = np.meshgrid(
                np.linspace(0, 1, grid.shape[0]),
                np.linspace(0, 1, grid.shape[1]),
                indexing="ij",
            )

            r_sq = (x - x0) ** 2 + (y - y0) ** 2
            data = strength * np.exp(-r_sq / (2 * radius**2))
            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

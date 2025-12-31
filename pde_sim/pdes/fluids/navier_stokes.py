"""2D Navier-Stokes in vorticity-stream function formulation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("navier-stokes")
class NavierStokesPDE(MultiFieldPDEPreset):
    """2D Navier-Stokes in vorticity-stream function formulation.

    Full vorticity equation with advection:

        dw/dt = nu * laplace(w) - u * d_dx(w) - v * d_dy(w)

    where velocity (u, v) is derived from stream function psi.

    Simplified version with explicit velocity fields:
        dw/dt = nu * laplace(w) - (d_dy(psi) * d_dx(w) - d_dx(psi) * d_dy(w))
        laplace(psi) = -w

    For simulation, we use a simplified model where we solve the
    vorticity transport with diffusion and a simple shear flow.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="navier-stokes",
            category="fluids",
            description="2D Navier-Stokes vorticity equation",
            equations={
                "w": "nu * laplace(w) - ux * d_dx(w) - uy * d_dy(w)",
            },
            parameters=[
                PDEParameter(
                    name="nu",
                    default=0.01,
                    description="Kinematic viscosity",
                    min_value=0.001,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="ux",
                    default=0.1,
                    description="Background velocity x",
                    min_value=-2.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="uy",
                    default=0.0,
                    description="Background velocity y",
                    min_value=-2.0,
                    max_value=2.0,
                ),
            ],
            num_fields=1,
            field_names=["w"],
            reference="2D incompressible Navier-Stokes",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        nu = parameters.get("nu", 0.01)
        ux = parameters.get("ux", 0.1)
        uy = parameters.get("uy", 0.0)

        # Vorticity transport with background flow
        rhs = f"{nu} * laplace(w)"
        if ux != 0:
            rhs += f" - {ux} * d_dx(w)"
        if uy != 0:
            rhs += f" - {uy} * d_dy(w)"

        return PDE(
            rhs={"w": rhs},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial vorticity distribution."""
        if ic_type in ("navier-stokes-default", "default", "vortex-pair"):
            # Counter-rotating vortex pair
            x0_1, y0_1 = ic_params.get("x1", 0.35), ic_params.get("y1", 0.5)
            x0_2, y0_2 = ic_params.get("x2", 0.65), ic_params.get("y2", 0.5)
            strength = ic_params.get("strength", 10.0)
            radius = ic_params.get("radius", 0.1)

            x, y = np.meshgrid(
                np.linspace(0, 1, grid.shape[0]),
                np.linspace(0, 1, grid.shape[1]),
                indexing="ij",
            )

            r1_sq = (x - x0_1) ** 2 + (y - y0_1) ** 2
            r2_sq = (x - x0_2) ** 2 + (y - y0_2) ** 2

            w1 = strength * np.exp(-r1_sq / (2 * radius**2))
            w2 = -strength * np.exp(-r2_sq / (2 * radius**2))

            data = w1 + w2
            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

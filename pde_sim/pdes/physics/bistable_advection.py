"""Bistable Allen-Cahn equation with advection."""

import math
from typing import Any

from pde import PDE, CartesianGrid

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("bistable-advection")
class BistableAdvectionPDE(ScalarPDEPreset):
    """Bistable Allen-Cahn equation with flow term.

    Combines the bistable reaction (Allen-Cahn with Allee effect) with
    directed advection:

        du/dt = D*laplace(u) + u*(u-a)*(1-u) + V*(cos(theta)*u_x + sin(theta)*u_y)

    where:
        - u is the population density (or order parameter)
        - D is the diffusion coefficient
        - a is the Allee threshold (unstable equilibrium between 0 and 1)
        - V is the flow velocity magnitude
        - theta is the flow direction in radians

    This models populations in flowing environments (e.g., rivers, ocean currents).
    The flow can assist or hinder invasion depending on its direction relative
    to the wave propagation direction.

    Key behaviors:
        - a < 0.5: Invasion front propagates in positive direction
        - a > 0.5: Invasion front propagates in negative direction
        - a = 0.5: Front is stationary (critical threshold)
        - Flow can push or pull the invasion front

    Reference:
        https://visualpde.com/reaction-diffusion/bistable-advection
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="bistable-advection",
            category="physics",
            description="Bistable Allen-Cahn equation with directed flow",
            equations={
                "u": "D*laplace(u) + u*(u-a)*(1-u) + V*(cos(theta)*d_dx(u) + sin(theta)*d_dy(u))",
            },
            parameters=[
                PDEParameter("D", "Diffusion coefficient"),
                PDEParameter("a", "Allee threshold (near critical at 0.5)"),
                PDEParameter("theta", "Flow direction in radians"),
                PDEParameter("V", "Flow velocity magnitude"),
            ],
            num_fields=1,
            field_names=["u"],
            reference="https://visualpde.com/reaction-diffusion/bistable-advection",
            supported_dimensions=[2],  # Currently 2D only (uses d_dy for advection)
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 0.02)
        a = parameters.get("a", 0.48)
        theta = parameters.get("theta", 1.8)
        V = parameters.get("V", 0.5)

        bc_spec = self._convert_bc(bc)

        # Build advection term: V*(cos(theta)*u_x + sin(theta)*u_y)
        vx = V * math.cos(theta)
        vy = V * math.sin(theta)

        # Build full equation
        # Bistable reaction: u*(u-a)*(1-u)
        # Diffusion: D*laplace(u)
        # Advection: vx*d_dx(u) + vy*d_dy(u)
        if V > 0:
            rhs = f"{D} * laplace(u) + u * (u - {a}) * (1 - u) + {vx} * d_dx(u) + {vy} * d_dy(u)"
        else:
            rhs = f"{D} * laplace(u) + u * (u - {a}) * (1 - u)"

        return PDE(
            rhs={"u": rhs},
            bc=bc_spec,
        )


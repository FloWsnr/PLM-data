"""Advection-diffusion equation."""

import math
from typing import Any

from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("advection")
class AdvectionPDE(ScalarPDEPreset):
    """Advection-diffusion equation.

    Based on visualpde.com advection equation:

        du/dt = D * laplace(u) + V * (advection term)

    Two velocity field types are supported (matching visual-pde sign convention):
    - Rotational: V*((y-L_y/2)*u_x - (x-L_x/2)*u_y) - counterclockwise for V>0
    - Directed: V*(cos(theta)*u_x + sin(theta)*u_y) - flow opposite to angle theta

    Parameters:
        D: Diffusion coefficient
        V: Advection velocity magnitude
        theta: Flow direction (radians, for directed flow)
        mode: 'rotational' or 'directed'
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="advection",
            category="basic",
            description="Advection-diffusion equation",
            equations={
                "u": "D * laplace(u) + V * (advection term)",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Diffusion coefficient",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="V",
                    default=0.10,
                    description="Advection velocity magnitude",
                    min_value=-5.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="theta",
                    default=-2.0,
                    description="Flow direction in radians (for directed mode)",
                    min_value=-6.4,
                    max_value=6.4,
                ),
                PDEParameter(
                    name="mode",
                    default=0,
                    description="0=rotational, 1=directed",
                    min_value=0,
                    max_value=1,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="https://visualpde.com/basic-pdes/advection-equation",
            supported_dimensions=[2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        V = parameters.get("V", 0.10)
        theta = parameters.get("theta", -2.0)
        mode = int(parameters.get("mode", 0))  # 0=rotational, 1=directed

        # Get domain size
        L_x = grid.axes_bounds[0][1] - grid.axes_bounds[0][0]
        L_y = grid.axes_bounds[1][1] - grid.axes_bounds[1][0]

        bc_spec = self._convert_bc(bc)

        if mode == 0:
            # Rotational velocity field (counterclockwise for positive V)
            # Matches visual-pde: V*((y-L_y/2)*u_x-(x-L_x/2)*u_y)
            half_y = L_y / 2
            half_x = L_x / 2
            advection = f"{V} * ((y - {half_y}) * d_dx(u) - (x - {half_x}) * d_dy(u))"
        else:
            # Directed velocity field: flow in direction theta
            # Matches visual-pde: V*(cos(theta)*u_x + sin(theta)*u_y)
            vx = V * math.cos(theta)
            vy = V * math.sin(theta)
            advection = f"{vx} * d_dx(u) + {vy} * d_dy(u)"

        # Build equation string
        if D > 0:
            rhs = f"{D} * laplace(u) + {advection}"
        else:
            rhs = advection

        return PDE(
            rhs={"u": rhs},
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> ScalarField:
        return create_initial_condition(grid, ic_type, ic_params)

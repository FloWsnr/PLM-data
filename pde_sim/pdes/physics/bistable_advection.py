"""Bistable Allen-Cahn equation with advection."""

import math
from typing import Any

from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

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
                PDEParameter(
                    name="D",
                    default=0.02,
                    description="Diffusion coefficient",
                    min_value=0.001,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="a",
                    default=0.48,
                    description="Allee threshold (near critical at 0.5)",
                    min_value=0.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="theta",
                    default=1.8,
                    description="Flow direction in radians",
                    min_value=-6.4,
                    max_value=6.4,
                ),
                PDEParameter(
                    name="V",
                    default=0.5,
                    description="Flow velocity magnitude",
                    min_value=0.0,
                    max_value=5.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="https://visualpde.com/reaction-diffusion/bistable-advection",
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

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> ScalarField:
        """Create initial state for bistable-advection equation.

        Default: Use step function to create an invasion front.
        """
        if ic_type in ("default", "bistable-advection-default"):
            # Create a step function initial condition
            # Left side at 1 (invaded), right side at 0 (uninvaded)
            ic_params = {
                "direction": ic_params.get("direction", "x"),
                "step_position": ic_params.get("step_position", 0.3),
                "value_low": ic_params.get("value_low", 0.0),
                "value_high": ic_params.get("value_high", 1.0),
                "smooth": ic_params.get("smooth", True),
                "width": ic_params.get("width", 0.05),
            }
            return create_initial_condition(grid, "step", ic_params)

        return create_initial_condition(grid, ic_type, ic_params)

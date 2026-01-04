"""Shallow water equations - real hyperbolic wave equations."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("shallow-water")
class ShallowWaterPDE(MultiFieldPDEPreset):
    """Linearized Shallow Water equations (full hyperbolic system).

    The shallow water equations describe free surface gravity waves:

        dh/dt = -H * (d_dx(u) + d_dy(v))
        du/dt = -g * d_dx(h)
        dv/dt = -g * d_dy(h)

    where:
        - h is the height perturbation from mean depth H
        - u, v are the depth-averaged velocity components
        - g is gravitational acceleration
        - H is the mean water depth

    The wave speed is c = sqrt(g*H).

    This is a hyperbolic system that supports propagating waves,
    unlike the diffusive approximation.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="shallow-water",
            category="fluids",
            description="Linearized shallow water wave equations",
            equations={
                "h": "-H * (d_dx(u) + d_dy(v))",
                "u": "-g * d_dx(h)",
                "v": "-g * d_dy(h)",
            },
            parameters=[
                PDEParameter(
                    name="g",
                    default=1.0,
                    description="Gravitational acceleration",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="H",
                    default=1.0,
                    description="Mean water depth",
                    min_value=0.1,
                    max_value=10.0,
                ),
            ],
            num_fields=3,
            field_names=["h", "u", "v"],
            reference="Shallow water wave equations (Saint-Venant)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        g = parameters.get("g", 1.0)
        H = parameters.get("H", 1.0)

        return PDE(
            rhs={
                "h": f"-{H} * (d_dx(u) + d_dy(v))",
                "u": f"-{g} * d_dx(h)",
                "v": f"-{g} * d_dy(h)",
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial water height perturbation with zero velocity."""
        # Get domain bounds from grid
        x_min, x_max = grid.axes_bounds[0]
        y_min, y_max = grid.axes_bounds[1]
        L_x = x_max - x_min
        L_y = y_max - y_min

        if ic_type in ("shallow-water-default", "drop", "gaussian-blobs"):
            # Water drop (Gaussian perturbation) - height only
            x0 = ic_params.get("x0", 0.5)
            y0 = ic_params.get("y0", 0.5)
            amplitude = ic_params.get("amplitude", 0.3)
            width = ic_params.get("width", 0.15)

            x, y = np.meshgrid(
                np.linspace(x_min, x_max, grid.shape[0]),
                np.linspace(y_min, y_max, grid.shape[1]),
                indexing="ij",
            )

            # Center coordinates (relative to domain)
            cx = x_min + x0 * L_x
            cy = y_min + y0 * L_y
            w = width * min(L_x, L_y)

            r_sq = (x - cx) ** 2 + (y - cy) ** 2
            h_data = amplitude * np.exp(-r_sq / (2 * w**2))

            # Zero initial velocities (drop starts at rest)
            u_data = np.zeros_like(h_data)
            v_data = np.zeros_like(h_data)

        elif ic_type == "ripple":
            # Multiple drops for interference patterns
            amplitude = ic_params.get("amplitude", 0.2)
            width = ic_params.get("width", 0.1)

            x, y = np.meshgrid(
                np.linspace(x_min, x_max, grid.shape[0]),
                np.linspace(y_min, y_max, grid.shape[1]),
                indexing="ij",
            )

            w = width * min(L_x, L_y)

            # Two drops at different positions
            cx1, cy1 = x_min + 0.3 * L_x, y_min + 0.5 * L_y
            cx2, cy2 = x_min + 0.7 * L_x, y_min + 0.5 * L_y

            r1_sq = (x - cx1) ** 2 + (y - cy1) ** 2
            r2_sq = (x - cx2) ** 2 + (y - cy2) ** 2

            h_data = amplitude * (
                np.exp(-r1_sq / (2 * w**2)) + np.exp(-r2_sq / (2 * w**2))
            )
            u_data = np.zeros_like(h_data)
            v_data = np.zeros_like(h_data)

        else:
            # Default: use standard IC generator for h, zeros for u, v
            h_field = create_initial_condition(grid, ic_type, ic_params)
            h_data = h_field.data
            u_data = np.zeros_like(h_data)
            v_data = np.zeros_like(h_data)

        h = ScalarField(grid, h_data)
        h.label = "h"
        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([h, u, v])

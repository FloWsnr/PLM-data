"""Shallow water equations - 2D wave propagation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("shallow-water")
class ShallowWaterPDE(MultiFieldPDEPreset):
    """2D Shallow Water Equations.

    The shallow water equations model water waves and ripples in a fluid
    layer where horizontal length scales are much larger than vertical depth.

    Water height evolution:
        dh/dt = -(d_dx(u) + d_dy(v)) * (h + H_e) - (d_dx(h) * u + d_dy(h) * v) - epsilon * h

    Horizontal momentum (x-direction):
        du/dt = nu * laplace(u) - g * d_dx(h) - k * u - u * d_dx(u) - v * d_dy(u) + f * v

    Horizontal momentum (y-direction):
        dv/dt = nu * laplace(v) - g * d_dy(h) - k * v - u * d_dx(v) - v * d_dy(v) - f * u

    where:
        - h is height perturbation from mean depth
        - u, v are horizontal velocity components
        - H_e is equilibrium water depth
        - g is gravitational acceleration
        - f is Coriolis parameter
        - k is linear drag coefficient
        - nu is kinematic viscosity
        - epsilon is height dissipation rate
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="shallow-water",
            category="fluids",
            description="2D shallow water wave equations",
            equations={
                "h": "-(d_dx(u) + d_dy(v)) * (h + H_e) - (d_dx(h) * u + d_dy(h) * v) - epsilon * h",
                "u": "nu * laplace(u) - g * d_dx(h) - k * u - u * d_dx(u) - v * d_dy(u) + f * v",
                "v": "nu * laplace(v) - g * d_dy(h) - k * v - u * d_dx(v) - v * d_dy(v) - f * u",
            },
            parameters=[
                PDEParameter(
                    name="H_e",
                    default=1.0,
                    description="Equilibrium water depth",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="g",
                    default=9.81,
                    description="Gravitational acceleration",
                    min_value=1.0,
                    max_value=20.0,
                ),
                PDEParameter(
                    name="f",
                    default=0.01,
                    description="Coriolis parameter",
                    min_value=0.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="k",
                    default=0.001,
                    description="Linear drag coefficient",
                    min_value=0.0,
                    max_value=0.1,
                ),
                PDEParameter(
                    name="nu",
                    default=0.5,
                    description="Kinematic viscosity",
                    min_value=0.01,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="epsilon",
                    default=0.0001,
                    description="Height dissipation rate",
                    min_value=0.0,
                    max_value=0.01,
                ),
            ],
            num_fields=3,
            field_names=["h", "u", "v"],
            reference="Shallow water equations (Saint-Venant)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        H_e = parameters.get("H_e", 1.0)
        g = parameters.get("g", 9.81)
        f = parameters.get("f", 0.01)
        k = parameters.get("k", 0.001)
        nu = parameters.get("nu", 0.5)
        epsilon = parameters.get("epsilon", 0.0001)

        return PDE(
            rhs={
                "h": f"-(d_dx(u) + d_dy(v)) * (h + {H_e}) - (d_dx(h) * u + d_dy(h) * v) - {epsilon} * h",
                "u": f"{nu} * laplace(u) - {g} * d_dx(h) - {k} * u - u * d_dx(u) - v * d_dy(u) + {f} * v",
                "v": f"{nu} * laplace(v) - {g} * d_dy(h) - {k} * v - u * d_dx(v) - v * d_dy(v) - {f} * u",
            },
            **self._get_pde_bc_kwargs(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
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

        elif ic_type == "dam-break":
            # Dam break initial condition (step function using tanh)
            # Reference: Visual PDE ShallowWaterEqnsDamBreaking preset
            # Uses 0.05*(1+tanh(x-L_x/2)) - step goes UP left-to-right
            step_width = ic_params.get("step_width", 0.05)
            amplitude = ic_params.get("amplitude", 0.05)  # Match Visual PDE default

            x, y = np.meshgrid(
                np.linspace(x_min, x_max, grid.shape[0]),
                np.linspace(y_min, y_max, grid.shape[1]),
                indexing="ij",
            )

            # Center coordinate
            cx = x_min + 0.5 * L_x

            # Tanh step function centered at x = L_x/2, matches Visual PDE
            h_data = amplitude * (1.0 + np.tanh((x - cx) / (step_width * L_x)))
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

        elif ic_type == "geostrophic-vortex":
            # Geostrophically balanced vortex for strong Coriolis (f ~ 1)
            # In geostrophic balance: f*v = g*dh/dx, f*u = -g*dh/dy
            # Reference: Visual PDE ShallowWaterEqnsVorticalSolitons preset
            x0 = ic_params.get("x0", 0.5)
            y0 = ic_params.get("y0", 0.5)
            amplitude = ic_params.get("amplitude", 0.02)
            radius = ic_params.get("radius", 0.15)
            f = ic_params.get("f", 1.0)
            g = ic_params.get("g", 9.81)

            x, y = np.meshgrid(
                np.linspace(x_min, x_max, grid.shape[0]),
                np.linspace(y_min, y_max, grid.shape[1]),
                indexing="ij",
            )

            # Normalize to [0, 1] for position calculations
            x_norm = (x - x_min) / L_x
            y_norm = (y - y_min) / L_y

            # Gaussian vortex radius in normalized coordinates
            r_sq = (x_norm - x0) ** 2 + (y_norm - y0) ** 2
            sigma = radius

            # Height perturbation (Gaussian)
            h_data = amplitude * np.exp(-r_sq / (2 * sigma**2))

            # Geostrophic velocities: u = -(g/f) * dh/dy, v = (g/f) * dh/dx
            # dh/dx = h * (-2*(x-x0))/(2*sigma^2) = -h * (x-x0)/sigma^2
            dh_dx_norm = -h_data * (x_norm - x0) / (sigma**2)
            dh_dy_norm = -h_data * (y_norm - y0) / (sigma**2)

            # Convert normalized gradients to physical gradients
            dh_dx = dh_dx_norm / L_x
            dh_dy = dh_dy_norm / L_y

            # Geostrophic balance: avoid division by zero if f is small
            if abs(f) > 1e-6:
                u_data = -(g / f) * dh_dy
                v_data = (g / f) * dh_dx
            else:
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

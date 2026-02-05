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
                PDEParameter("H_e", "Equilibrium water depth"),
                PDEParameter("g", "Gravitational acceleration"),
                PDEParameter("f", "Coriolis parameter"),
                PDEParameter("k", "Linear drag coefficient"),
                PDEParameter("nu", "Kinematic viscosity"),
                PDEParameter("epsilon", "Height dissipation rate"),
            ],
            num_fields=3,
            field_names=["h", "u", "v"],
            reference="Shallow water equations (Saint-Venant)",
            supported_dimensions=[2],
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

        rng = np.random.default_rng(ic_params.get("seed"))

        if ic_type in ("shallow-water-default", "drop", "gaussian-blob"):
            # Water drop (Gaussian perturbation) - height only
            x0 = ic_params.get("x0")
            y0 = ic_params.get("y0")
            if x0 is None or x0 == "random" or y0 is None or y0 == "random":
                raise ValueError("drop requires x0 and y0 (or random)")
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
            step_width = ic_params.get("step_width", 0.05)
            amplitude = ic_params.get("amplitude", 0.05)

            x, y = np.meshgrid(
                np.linspace(x_min, x_max, grid.shape[0]),
                np.linspace(y_min, y_max, grid.shape[1]),
                indexing="ij",
            )

            # Center coordinate (randomize if not specified)
            dam_position = ic_params.get("dam_position")
            if dam_position is None or dam_position == "random":
                raise ValueError("dam-break requires dam_position (or random)")
            cx = x_min + dam_position * L_x

            # Tanh step function
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

            # Two drops at different positions (randomize if not specified)
            x1 = ic_params.get("x1")
            y1 = ic_params.get("y1")
            x2 = ic_params.get("x2")
            y2 = ic_params.get("y2")
            if (
                x1 is None or x1 == "random"
                or y1 is None or y1 == "random"
                or x2 is None or x2 == "random"
                or y2 is None or y2 == "random"
            ):
                raise ValueError("ripple requires x1, y1, x2, y2 (or random)")
            cx1, cy1 = x_min + x1 * L_x, y_min + y1 * L_y
            cx2, cy2 = x_min + x2 * L_x, y_min + y2 * L_y

            r1_sq = (x - cx1) ** 2 + (y - cy1) ** 2
            r2_sq = (x - cx2) ** 2 + (y - cy2) ** 2

            h_data = amplitude * (
                np.exp(-r1_sq / (2 * w**2)) + np.exp(-r2_sq / (2 * w**2))
            )
            u_data = np.zeros_like(h_data)
            v_data = np.zeros_like(h_data)

        elif ic_type == "geostrophic-vortex":
            # Geostrophically balanced vortex for strong Coriolis (f ~ 1)
            x0 = ic_params.get("x0")
            y0 = ic_params.get("y0")
            if x0 is None or x0 == "random" or y0 is None or y0 == "random":
                raise ValueError("geostrophic-vortex requires x0 and y0 (or random)")
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

    def get_position_params(self, ic_type: str) -> set[str]:
        if ic_type in ("shallow-water-default", "drop", "gaussian-blob"):
            return {"x0", "y0"}
        if ic_type == "dam-break":
            return {"dam_position"}
        if ic_type == "ripple":
            return {"x1", "y1", "x2", "y2"}
        if ic_type == "geostrophic-vortex":
            return {"x0", "y0"}
        return super().get_position_params(ic_type)

    def resolve_ic_params(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> dict[str, Any]:
        resolved = ic_params.copy()
        if ic_type in ("shallow-water-default", "drop", "gaussian-blob"):
            if "x0" not in resolved or "y0" not in resolved:
                raise ValueError("drop requires x0 and y0 (or random)")
            if resolved["x0"] == "random" or resolved["y0"] == "random":
                rng = np.random.default_rng(resolved.get("seed"))
                if resolved["x0"] == "random":
                    resolved["x0"] = rng.uniform(0.2, 0.8)
                if resolved["y0"] == "random":
                    resolved["y0"] = rng.uniform(0.2, 0.8)
            if resolved["x0"] is None or resolved["y0"] is None:
                raise ValueError("drop requires x0 and y0 (or random)")
            return resolved
        if ic_type == "dam-break":
            if "dam_position" not in resolved:
                raise ValueError("dam-break requires dam_position (or random)")
            if resolved["dam_position"] == "random":
                rng = np.random.default_rng(resolved.get("seed"))
                resolved["dam_position"] = rng.uniform(0.3, 0.7)
            if resolved["dam_position"] is None:
                raise ValueError("dam-break requires dam_position (or random)")
            return resolved
        if ic_type == "ripple":
            required = ("x1", "y1", "x2", "y2")
            for key in required:
                if key not in resolved:
                    raise ValueError("ripple requires x1, y1, x2, y2 (or random)")
            if any(resolved[key] == "random" for key in required):
                rng = np.random.default_rng(resolved.get("seed"))
                if resolved["x1"] == "random":
                    resolved["x1"] = rng.uniform(0.15, 0.4)
                if resolved["y1"] == "random":
                    resolved["y1"] = rng.uniform(0.3, 0.7)
                if resolved["x2"] == "random":
                    resolved["x2"] = rng.uniform(0.6, 0.85)
                if resolved["y2"] == "random":
                    resolved["y2"] = rng.uniform(0.3, 0.7)
            if any(resolved[key] is None for key in required):
                raise ValueError("ripple requires x1, y1, x2, y2 (or random)")
            return resolved
        if ic_type == "geostrophic-vortex":
            if "x0" not in resolved or "y0" not in resolved:
                raise ValueError("geostrophic-vortex requires x0 and y0 (or random)")
            if resolved["x0"] == "random" or resolved["y0"] == "random":
                rng = np.random.default_rng(resolved.get("seed"))
                if resolved["x0"] == "random":
                    resolved["x0"] = rng.uniform(0.2, 0.8)
                if resolved["y0"] == "random":
                    resolved["y0"] = rng.uniform(0.2, 0.8)
            if resolved["x0"] is None or resolved["y0"] is None:
                raise ValueError("geostrophic-vortex requires x0 and y0 (or random)")
            return resolved
        return super().resolve_ic_params(grid, ic_type, ic_params)

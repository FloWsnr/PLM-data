"""Zakharov-Kuznetsov equation - 2D generalization of KdV for vortical solitons."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import PDEMetadata, PDEParameter, ScalarPDEPreset
from .. import register_pde


@register_pde("zakharov-kuznetsov")
class ZakharovKuznetsovPDE(ScalarPDEPreset):
    """Zakharov-Kuznetsov equation - 2D generalization of KdV.

    Based on visualpde.com formulation:

        du/dt = -d³u/dx³ - d³u/(dx dy²) - u * du/dx - b * ∇⁴u

    The 2D analog of the KdV equation supporting vortical solitons.
    The mixed derivative term d³u/(dx dy²) enables true 2D dynamics.
    The biharmonic dissipation term b∇⁴u reduces radiation.

    Key properties:
        - 2D generalization of KdV equation
        - Supports vortical (radially symmetric) solitons
        - Solitons travel in the positive x direction
        - Small dissipation (b) helps stabilize the solution

    Initial condition (vortical soliton):
        u(x,y,0) = sech²(a * (x² + y²))

    where a controls the soliton width.

    Reference: Zakharov & Kuznetsov (1974), visualpde.com
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="zakharov-kuznetsov",
            category="physics",
            description="2D Zakharov-Kuznetsov equation for vortical solitons",
            equations={
                "u": "-d_dx(laplace(u)) - u * d_dx(u) - b * laplace(laplace(u))",
            },
            parameters=[
                PDEParameter("b", "Biharmonic dissipation coefficient"),
                PDEParameter("theta", "Propagation angle in radians (0 = +x direction)"),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Zakharov & Kuznetsov (1974)",
            supported_dimensions=[2],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Zakharov-Kuznetsov equation PDE.

        Args:
            parameters: Dictionary with dissipation coefficient b and theta.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        b = parameters.get("b", 0.008)
        theta = parameters.get("theta", 0.0)

        # Zakharov-Kuznetsov equation with rotated propagation direction:
        # du/dt = -d_s(laplace(u)) - u*d_s(u) - b*∇⁴u
        #
        # where d_s is the directional derivative in the propagation direction:
        #   d_s = cos(θ)*d_dx + sin(θ)*d_dy
        #
        # For theta=0, this reduces to the standard ZK equation with +x propagation.
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Build directional derivative expression
        # d_s(f) = cos(θ)*d_dx(f) + sin(θ)*d_dy(f)
        if abs(sin_t) < 1e-10:
            # Pure x-direction (theta ≈ 0 or π)
            sign = 1.0 if cos_t > 0 else -1.0
            rhs = f"{sign} * (-d_dx(laplace(u)) - u * d_dx(u)) - {b} * laplace(laplace(u))"
        elif abs(cos_t) < 1e-10:
            # Pure y-direction (theta ≈ ±π/2)
            sign = 1.0 if sin_t > 0 else -1.0
            rhs = f"{sign} * (-d_dy(laplace(u)) - u * d_dy(u)) - {b} * laplace(laplace(u))"
        else:
            # General direction
            rhs = (
                f"-{cos_t} * d_dx(laplace(u)) - {sin_t} * d_dy(laplace(u)) "
                f"- u * ({cos_t} * d_dx(u) + {sin_t} * d_dy(u)) "
                f"- {b} * laplace(laplace(u))"
            )

        return PDE(
            rhs={"u": rhs},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> ScalarField:
        """Create initial state - vortical soliton for 2D dynamics.

        Default: Radially symmetric vortical soliton centered at origin.
        """
        if ic_type in ("zakharov-kuznetsov-default", "default", "vortical-soliton"):
            # Vortical soliton: u = sech²(a * r²) where r² = x² + y²
            # From visual-pde: a = 0.06 for domain scale 350
            a = ic_params.get("a", 0.06)

            x_bounds = grid.axes_bounds[0]
            y_bounds = grid.axes_bounds[1]

            # Center the soliton (domain centered at origin in visual-pde)
            x_center = ic_params.get("x_center")
            y_center = ic_params.get("y_center")
            if (
                x_center is None or x_center == "random"
                or y_center is None or y_center == "random"
            ):
                raise ValueError("zakharov-kuznetsov vortical-soliton requires x_center and y_center (or random)")

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            # Radial distance squared from center
            r_sq = (X - x_center) ** 2 + (Y - y_center) ** 2

            # Vortical soliton: sech²(a * r²)
            data = 1.0 / np.cosh(a * r_sq) ** 2

            return ScalarField(grid, data)

        if ic_type == "offset-soliton":
            # Soliton offset from center to show propagation
            a = ic_params.get("a", 0.06)

            x_bounds = grid.axes_bounds[0]
            y_bounds = grid.axes_bounds[1]

            x_center = ic_params.get("x_center")
            y_center = ic_params.get("y_center")
            if (
                x_center is None or x_center == "random"
                or y_center is None or y_center == "random"
            ):
                raise ValueError("zakharov-kuznetsov offset-soliton requires x_center and y_center (or random)")

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            r_sq = (X - x_center) ** 2 + (Y - y_center) ** 2
            data = 1.0 / np.cosh(a * r_sq) ** 2

            return ScalarField(grid, data)

        if ic_type == "two-solitons":
            # Two solitons to observe interaction/collision dynamics
            a1 = ic_params.get("a1", 0.01)
            a2 = ic_params.get("a2", 0.01)

            x_bounds = grid.axes_bounds[0]
            y_bounds = grid.axes_bounds[1]

            # First soliton: left side
            x1 = ic_params.get("x1")
            y1 = ic_params.get("y1")

            # Second soliton: also left but different y position
            x2 = ic_params.get("x2")
            y2 = ic_params.get("y2")
            if (
                x1 is None or x1 == "random"
                or y1 is None or y1 == "random"
                or x2 is None or x2 == "random"
                or y2 is None or y2 == "random"
            ):
                raise ValueError("zakharov-kuznetsov two-solitons requires x1, y1, x2, y2 (or random)")

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            r1_sq = (X - x1) ** 2 + (Y - y1) ** 2
            r2_sq = (X - x2) ** 2 + (Y - y2) ** 2

            data = 1.0 / np.cosh(a1 * r1_sq) ** 2 + 1.0 / np.cosh(a2 * r2_sq) ** 2

            return ScalarField(grid, data)

        if ic_type == "random-solitons":
            # N randomly placed solitons with random widths
            # Parameters:
            #   n: number of solitons (default 3)
            #   a_min, a_max: range for width parameter (default 0.005-0.015)
            #   margin: fraction of domain to avoid at edges (default 0.1)
            x_bounds = grid.axes_bounds[0]
            y_bounds = grid.axes_bounds[1]
            Lx = x_bounds[1] - x_bounds[0]
            Ly = y_bounds[1] - y_bounds[0]

            # Create coordinate arrays
            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            positions = ic_params.get("positions")
            a_values = ic_params.get("a_values")
            if (
                positions is None or positions == "random"
                or a_values is None or a_values == "random"
            ):
                raise ValueError("zakharov-kuznetsov random-solitons requires positions and a_values (or random)")

            data = np.zeros_like(X)
            for (xi, yi), ai in zip(positions, a_values):
                r_sq = (X - xi) ** 2 + (Y - yi) ** 2
                data += 1.0 / np.cosh(ai * r_sq) ** 2

            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

    def resolve_ic_params(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> dict[str, Any]:
        if ic_type in ("zakharov-kuznetsov-default", "default", "vortical-soliton"):
            resolved = ic_params.copy()
            required = ["x_center", "y_center"]
            for key in required:
                if key not in resolved:
                    raise ValueError("zakharov-kuznetsov vortical-soliton requires x_center and y_center (or random)")
            if any(resolved[key] == "random" for key in required):
                rng = np.random.default_rng(resolved.get("seed"))
                x_bounds = grid.axes_bounds[0]
                y_bounds = grid.axes_bounds[1]
                if resolved["x_center"] == "random":
                    resolved["x_center"] = rng.uniform(x_bounds[0], x_bounds[1])
                if resolved["y_center"] == "random":
                    resolved["y_center"] = rng.uniform(y_bounds[0], y_bounds[1])
            if any(resolved[key] is None or resolved[key] == "random" for key in required):
                raise ValueError("zakharov-kuznetsov vortical-soliton requires x_center and y_center (or random)")
            return resolved

        if ic_type == "offset-soliton":
            resolved = ic_params.copy()
            required = ["x_center", "y_center"]
            for key in required:
                if key not in resolved:
                    raise ValueError("zakharov-kuznetsov offset-soliton requires x_center and y_center (or random)")
            if any(resolved[key] == "random" for key in required):
                rng = np.random.default_rng(resolved.get("seed"))
                x_bounds = grid.axes_bounds[0]
                y_bounds = grid.axes_bounds[1]
                if resolved["x_center"] == "random":
                    resolved["x_center"] = rng.uniform(x_bounds[0], x_bounds[1])
                if resolved["y_center"] == "random":
                    resolved["y_center"] = rng.uniform(y_bounds[0], y_bounds[1])
            if any(resolved[key] is None or resolved[key] == "random" for key in required):
                raise ValueError("zakharov-kuznetsov offset-soliton requires x_center and y_center (or random)")
            return resolved

        if ic_type == "two-solitons":
            resolved = ic_params.copy()
            required = ["x1", "y1", "x2", "y2"]
            for key in required:
                if key not in resolved:
                    raise ValueError("zakharov-kuznetsov two-solitons requires x1, y1, x2, y2 (or random)")
            if any(resolved[key] == "random" for key in required):
                rng = np.random.default_rng(resolved.get("seed"))
                x_bounds = grid.axes_bounds[0]
                y_bounds = grid.axes_bounds[1]
                if resolved["x1"] == "random":
                    resolved["x1"] = rng.uniform(x_bounds[0], x_bounds[1])
                if resolved["y1"] == "random":
                    resolved["y1"] = rng.uniform(y_bounds[0], y_bounds[1])
                if resolved["x2"] == "random":
                    resolved["x2"] = rng.uniform(x_bounds[0], x_bounds[1])
                if resolved["y2"] == "random":
                    resolved["y2"] = rng.uniform(y_bounds[0], y_bounds[1])
            if any(resolved[key] is None or resolved[key] == "random" for key in required):
                raise ValueError("zakharov-kuznetsov two-solitons requires x1, y1, x2, y2 (or random)")
            return resolved

        if ic_type == "random-solitons":
            resolved = ic_params.copy()
            if "positions" not in resolved or "a_values" not in resolved:
                raise ValueError("zakharov-kuznetsov random-solitons requires positions and a_values (or random)")
            if resolved["positions"] == "random" or resolved["a_values"] == "random":
                if "n" not in resolved or "a_min" not in resolved or "a_max" not in resolved or "margin" not in resolved:
                    raise ValueError("zakharov-kuznetsov random-solitons random generation requires n, a_min, a_max, margin")
                rng = np.random.default_rng(resolved.get("seed"))
                x_bounds = grid.axes_bounds[0]
                y_bounds = grid.axes_bounds[1]
                Lx = x_bounds[1] - x_bounds[0]
                Ly = y_bounds[1] - y_bounds[0]
                x_range = (
                    x_bounds[0] + resolved["margin"] * Lx,
                    x_bounds[0] + (1 - resolved["margin"]) * Lx,
                )
                y_range = (
                    y_bounds[0] + resolved["margin"] * Ly,
                    y_bounds[0] + (1 - resolved["margin"]) * Ly,
                )
                positions = [
                    [rng.uniform(x_range[0], x_range[1]), rng.uniform(y_range[0], y_range[1])]
                    for _ in range(resolved["n"])
                ]
                a_values = [rng.uniform(resolved["a_min"], resolved["a_max"]) for _ in range(resolved["n"])]
                resolved["positions"] = positions
                resolved["a_values"] = a_values
            if resolved["positions"] is None or resolved["a_values"] is None:
                raise ValueError("zakharov-kuznetsov random-solitons requires positions and a_values (or random)")
            return resolved

        return super().resolve_ic_params(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        b = parameters.get("b", 0.008)
        theta = parameters.get("theta", 0.0)

        if abs(theta) < 1e-10:
            return {
                "u": f"-d_dx(laplace(u)) - u * d_dx(u) - {b} * laplace(laplace(u))",
            }
        else:
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            return {
                "u": (
                    f"-({cos_t:.3f}*d_dx + {sin_t:.3f}*d_dy)(laplace(u)) "
                    f"- u*({cos_t:.3f}*d_dx + {sin_t:.3f}*d_dy)(u) "
                    f"- {b}*laplace(laplace(u))"
                ),
            }

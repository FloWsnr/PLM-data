"""Zakharov-Kuznetsov equation - 2D generalization of KdV for vortical solitons."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from ..base import PDEMetadata, PDEParameter, ScalarPDEPreset
from .. import register_pde


@register_pde("zakharov-kuznetsov")
class ZakharovKuznetsovPDE(ScalarPDEPreset):
    """Zakharov-Kuznetsov equation - 2D generalization of KdV.

    Based on visualpde.com formulation:

        du/dt = -d³u/dx³ - d³u/(dx dy²) - u * du/dx - b * nabla⁴u

    The 2D analog of the KdV equation supporting vortical solitons.
    The mixed derivative term d³u/(dx dy²) enables true 2D dynamics.
    The biharmonic dissipation term b*nabla⁴u reduces radiation.

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
        b = parameters["b"]
        theta = parameters["theta"]

        # Zakharov-Kuznetsov equation with rotated propagation direction:
        # du/dt = -d_s(laplace(u)) - u*d_s(u) - b*nabla⁴u
        #
        # where d_s is the directional derivative in the propagation direction:
        #   d_s = cos(theta)*d_dx + sin(theta)*d_dy
        #
        # For theta=0, this reduces to the standard ZK equation with +x propagation.
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Build directional derivative expression
        # d_s(f) = cos(theta)*d_dx(f) + sin(theta)*d_dy(f)
        if abs(sin_t) < 1e-10:
            # Pure x-direction (theta ~ 0 or pi)
            sign = 1.0 if cos_t > 0 else -1.0
            rhs = f"{sign} * (-d_dx(laplace(u)) - u * d_dx(u)) - {b} * laplace(laplace(u))"
        elif abs(cos_t) < 1e-10:
            # Pure y-direction (theta ~ +/-pi/2)
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
        randomize = kwargs.get("randomize", False)

        if ic_type in ("zakharov-kuznetsov-default", "default", "vortical-soliton"):
            # Vortical soliton: u = sech²(a * r²) where r² = x² + y²
            # From visual-pde: a = 0.06 for domain scale 350
            a = ic_params["a"]

            x_bounds = grid.axes_bounds[0]
            y_bounds = grid.axes_bounds[1]

            # Center the soliton (domain centered at origin in visual-pde)
            if randomize:
                rng = np.random.default_rng(ic_params.get("seed"))
                ic_params["x_center"] = rng.uniform(x_bounds[0], x_bounds[1])
                ic_params["y_center"] = rng.uniform(y_bounds[0], y_bounds[1])

            x_center = ic_params["x_center"]
            y_center = ic_params["y_center"]
            if x_center is None or y_center is None:
                raise ValueError("zakharov-kuznetsov vortical-soliton requires x_center and y_center")

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
            a = ic_params["a"]

            x_bounds = grid.axes_bounds[0]
            y_bounds = grid.axes_bounds[1]

            if randomize:
                rng = np.random.default_rng(ic_params.get("seed"))
                ic_params["x_center"] = rng.uniform(x_bounds[0], x_bounds[1])
                ic_params["y_center"] = rng.uniform(y_bounds[0], y_bounds[1])

            x_center = ic_params["x_center"]
            y_center = ic_params["y_center"]
            if x_center is None or y_center is None:
                raise ValueError("zakharov-kuznetsov offset-soliton requires x_center and y_center")

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            r_sq = (X - x_center) ** 2 + (Y - y_center) ** 2
            data = 1.0 / np.cosh(a * r_sq) ** 2

            return ScalarField(grid, data)

        if ic_type == "two-solitons":
            # Two solitons to observe interaction/collision dynamics
            a1 = ic_params["a1"]
            a2 = ic_params["a2"]

            x_bounds = grid.axes_bounds[0]
            y_bounds = grid.axes_bounds[1]

            if randomize:
                rng = np.random.default_rng(ic_params.get("seed"))
                ic_params["x1"] = rng.uniform(x_bounds[0], x_bounds[1])
                ic_params["y1"] = rng.uniform(y_bounds[0], y_bounds[1])
                ic_params["x2"] = rng.uniform(x_bounds[0], x_bounds[1])
                ic_params["y2"] = rng.uniform(y_bounds[0], y_bounds[1])

            # First soliton: left side
            x1 = ic_params["x1"]
            y1 = ic_params["y1"]

            # Second soliton: also left but different y position
            x2 = ic_params["x2"]
            y2 = ic_params["y2"]
            if x1 is None or y1 is None or x2 is None or y2 is None:
                raise ValueError("zakharov-kuznetsov two-solitons requires x1, y1, x2, y2")

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            r1_sq = (X - x1) ** 2 + (Y - y1) ** 2
            r2_sq = (X - x2) ** 2 + (Y - y2) ** 2

            data = 1.0 / np.cosh(a1 * r1_sq) ** 2 + 1.0 / np.cosh(a2 * r2_sq) ** 2

            return ScalarField(grid, data)

        if ic_type == "random-solitons":
            # N randomly placed solitons with random widths
            x_bounds = grid.axes_bounds[0]
            y_bounds = grid.axes_bounds[1]
            Lx = x_bounds[1] - x_bounds[0]
            Ly = y_bounds[1] - y_bounds[0]

            # Create coordinate arrays
            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            if randomize:
                if "n" not in ic_params or "a_min" not in ic_params or "a_max" not in ic_params or "margin" not in ic_params:
                    raise ValueError("zakharov-kuznetsov random-solitons random generation requires n, a_min, a_max, margin")
                rng = np.random.default_rng(ic_params.get("seed"))
                x_range = (
                    x_bounds[0] + ic_params["margin"] * Lx,
                    x_bounds[0] + (1 - ic_params["margin"]) * Lx,
                )
                y_range = (
                    y_bounds[0] + ic_params["margin"] * Ly,
                    y_bounds[0] + (1 - ic_params["margin"]) * Ly,
                )
                positions = [
                    [rng.uniform(x_range[0], x_range[1]), rng.uniform(y_range[0], y_range[1])]
                    for _ in range(ic_params["n"])
                ]
                a_values = [rng.uniform(ic_params["a_min"], ic_params["a_max"]) for _ in range(ic_params["n"])]
                ic_params["positions"] = positions
                ic_params["a_values"] = a_values

            positions = ic_params["positions"]
            a_values = ic_params["a_values"]
            if positions is None or a_values is None:
                raise ValueError("zakharov-kuznetsov random-solitons requires positions and a_values")

            data = np.zeros_like(X)
            for (xi, yi), ai in zip(positions, a_values):
                r_sq = (X - xi) ** 2 + (Y - yi) ** 2
                data += 1.0 / np.cosh(ai * r_sq) ** 2

            return ScalarField(grid, data)

        return super().create_initial_state(grid, ic_type, ic_params, **kwargs)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        b = parameters["b"]
        theta = parameters["theta"]

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

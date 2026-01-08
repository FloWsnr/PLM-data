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
                PDEParameter(
                    name="b",
                    default=0.008,
                    description="Biharmonic dissipation coefficient",
                    min_value=0.0,
                    max_value=0.1,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Zakharov & Kuznetsov (1974)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Zakharov-Kuznetsov equation PDE.

        Args:
            parameters: Dictionary with dissipation coefficient b.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        b = parameters.get("b", 0.008)

        # Zakharov-Kuznetsov equation (simplified form):
        # du/dt = -d_dx(laplace(u)) - u*d_dx(u) - b*∇⁴u
        #
        # This uses the identity:
        #   -∂³u/∂x³ - ∂³u/(∂x∂y²) = -∂/∂x(∂²u/∂x² + ∂²u/∂y²) = -d_dx(laplace(u))
        #
        # This formulation is more numerically stable than nested first derivatives
        # because we compute the Laplacian (2nd derivatives) first, then one 1st derivative.
        return PDE(
            rhs={
                "u": f"-d_dx(laplace(u)) - u * d_dx(u) - {b} * laplace(laplace(u))"
            },
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
            x_center = ic_params.get("x_center", (x_bounds[0] + x_bounds[1]) / 2)
            y_center = ic_params.get("y_center", (y_bounds[0] + y_bounds[1]) / 2)

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
            Lx = x_bounds[1] - x_bounds[0]

            # Offset to the left so we can see it move right
            x_center = ic_params.get("x_center", x_bounds[0] + Lx * 0.3)
            y_center = ic_params.get("y_center", (y_bounds[0] + y_bounds[1]) / 2)

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            r_sq = (X - x_center) ** 2 + (Y - y_center) ** 2
            data = 1.0 / np.cosh(a * r_sq) ** 2

            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        b = parameters.get("b", 0.008)
        return {
            "u": f"-d_dx(laplace(u)) - u * d_dx(u) - {b} * laplace(laplace(u))",
        }

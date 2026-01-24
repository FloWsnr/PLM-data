"""Perona-Malik anisotropic diffusion for edge-preserving smoothing."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("perona-malik")
class PeronaMalikPDE(ScalarPDEPreset):
    """Perona-Malik anisotropic diffusion equation.

    Based on visualpde.com formulation:

        du/dt = div(g(|grad(u)|) * grad(u))

    where g(s) is an edge-stopping function.

    Using exponential diffusivity: g(|grad(u)|) = exp(-D * |grad(u)|^2)

    The full divergence form expands as:
        div(g*grad(u)) = g*laplace(u) + grad(g)·grad(u)

    With g = exp(-D*v) where v = |grad(u)|^2 = u_x^2 + u_y^2:
        grad(g) = -2D*g * (u_x*u_xx + u_y*u_xy, u_x*u_xy + u_y*u_yy)
        grad(g)·grad(u) = -2D*g * [u_x^2*u_xx + 2*u_x*u_y*u_xy + u_y^2*u_yy]

    Full equation:
        du/dt = g * [laplace(u) - 2D*(u_x^2*u_xx + 2*u_x*u_y*u_xy + u_y^2*u_yy)]

    Key features:
        - Edge-preserving: smooths homogeneous regions while preserving edges
        - Edge-sharpening: the cross-term enhances edges over time
        - Adaptive diffusion: diffusion coefficient depends on local gradient
        - Image denoising: removes noise while maintaining structure

    Applications:
        - Image denoising
        - Medical imaging (CT/MRI enhancement)
        - Computer vision preprocessing
        - Segmentation

    Note: The original formulation is mathematically ill-posed, but
    discretization provides implicit regularization.

    Reference: Perona & Malik (1990), Weickert (1998)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="perona-malik",
            category="physics",
            description="Perona-Malik edge-preserving anisotropic diffusion",
            equations={
                "u": "div(exp(-D * |grad(u)|^2) * grad(u))",
            },
            parameters=[
                PDEParameter("D", "Edge sensitivity (higher = more edge preservation)"),
                PDEParameter("sigma", "Initial noise level (for initial condition)"),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Perona & Malik (1990) Scale-space and edge detection",
            supported_dimensions=[2],  # Currently 2D only (uses d_dy, u_xy cross-terms)
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Perona-Malik PDE with full divergence form.

        Args:
            parameters: Dictionary with D.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        D = parameters.get("D", 5.0)

        # Full Perona-Malik equation:
        # du/dt = div(g * grad(u)) where g = exp(-D * |grad(u)|^2)
        #
        # Expanded:
        # div(g * grad(u)) = g * laplace(u) + grad(g) · grad(u)
        #
        # With g = exp(-D * gradient_squared(u)):
        # grad(g) = -2D * g * (u_x*u_xx + u_y*u_xy, u_x*u_xy + u_y*u_yy)
        # grad(g) · grad(u) = -2D * g * [u_x^2*u_xx + 2*u_x*u_y*u_xy + u_y^2*u_yy]
        #
        # Full RHS:
        # g * [laplace(u) - 2D * (u_x^2*u_xx + 2*u_x*u_y*u_xy + u_y^2*u_yy)]

        # Define the diffusivity g
        g = f"exp(-{D} * gradient_squared(u))"

        # Define partial derivatives
        u_x = "d_dx(u)"
        u_y = "d_dy(u)"
        u_xx = "d_dx(d_dx(u))"
        u_yy = "d_dy(d_dy(u))"
        u_xy = "d_dx(d_dy(u))"

        # The cross-term: u_x^2*u_xx + 2*u_x*u_y*u_xy + u_y^2*u_yy
        cross_term = (
            f"({u_x})**2 * ({u_xx}) + "
            f"2 * ({u_x}) * ({u_y}) * ({u_xy}) + "
            f"({u_y})**2 * ({u_yy})"
        )

        # Full RHS: g * [laplace(u) - 2D * cross_term]
        rhs = f"({g}) * (laplace(u) - 2 * {D} * ({cross_term}))"

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
        """Create initial image-like pattern with edges and noise.

        Default: step pattern with added noise (like noisy image with edges).
        """
        if ic_type in ("perona-malik-default", "default"):
            sigma = ic_params.get("sigma", 1.0)
            seed = ic_params.get("seed")
            if seed is not None:
                np.random.seed(seed)

            x_bounds = grid.axes_bounds[0]
            y_bounds = grid.axes_bounds[1]
            Lx = x_bounds[1] - x_bounds[0]
            Ly = y_bounds[1] - y_bounds[0]

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            # Normalize to [0,1]
            X_norm = (X - x_bounds[0]) / Lx
            Y_norm = (Y - y_bounds[0]) / Ly

            # Step pattern with edges
            data = np.zeros(grid.shape)
            data[X_norm > 0.3] = 0.5
            data[X_norm > 0.7] = 1.0
            data[Y_norm > 0.5] += 0.3

            # Add noise
            noise = sigma * 0.3 * (np.random.rand(*grid.shape) - 0.5)
            data += noise

            return ScalarField(grid, data)

        if ic_type == "text-image":
            # Text-like pattern
            seed = ic_params.get("seed")
            if seed is not None:
                np.random.seed(seed)
            sigma = ic_params.get("sigma", 1.0)

            # Create simple pattern
            x_bounds = grid.axes_bounds[0]
            y_bounds = grid.axes_bounds[1]
            Lx = x_bounds[1] - x_bounds[0]
            Ly = y_bounds[1] - y_bounds[0]

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            # Create stripes and shapes
            data = np.sin(4 * np.pi * X / Lx) * np.sin(4 * np.pi * Y / Ly)
            data = (data > 0).astype(float)

            # Add noise
            data += sigma * 0.1 * np.random.randn(*grid.shape)

            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        D = parameters.get("D", 5.0)
        return {"u": f"div(exp(-{D} * |grad(u)|^2) * grad(u))"}

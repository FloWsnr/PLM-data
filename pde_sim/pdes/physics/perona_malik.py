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

    Approximated as:
        du/dt = laplace(u) * exp(-D * gradient_squared(u))

    Key features:
        - Edge-preserving: smooths homogeneous regions while preserving edges
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
                "u": "laplace(u) * exp(-D * gradient_squared(u))",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=5.0,
                    description="Edge sensitivity (higher = more edge preservation)",
                    min_value=0.1,
                    max_value=20.0,
                ),
                PDEParameter(
                    name="sigma",
                    default=1.0,
                    description="Initial noise level (for initial condition)",
                    min_value=0.0,
                    max_value=2.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Perona & Malik (1990) Scale-space and edge detection",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Perona-Malik PDE.

        Args:
            parameters: Dictionary with D.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        D = parameters.get("D", 5.0)

        # Perona-Malik with exponential diffusivity:
        # du/dt = laplace(u) * exp(-D * |grad(u)|^2)
        return PDE(
            rhs={"u": f"laplace(u) * exp(-{D} * gradient_squared(u))"},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
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
        return {"u": f"laplace(u) * exp(-{D} * gradient_squared(u))"}

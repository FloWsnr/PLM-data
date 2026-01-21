"""Fokker-Planck equation for probability density evolution."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("fokker-planck")
class FokkerPlanckPDE(ScalarPDEPreset):
    """Fokker-Planck equation for probability density evolution.

    The Fokker-Planck equation (also known as Kolmogorov forward equation)
    describes the time evolution of a probability density function under
    the influence of drift and diffusion.

    General form:
        dp/dt = -div(mu * p) + div(D * grad(p))

    In 2D with constant diffusion and harmonic potential drift:
        dp/dt = -d/dx(mu_x * p) - d/dy(mu_y * p) + D * laplace(p)

    where:
        - p(x, y, t) is the probability density
        - mu = (mu_x, mu_y) is the drift field (velocity toward equilibrium)
        - D is the diffusion coefficient

    The drift field can represent:
        - Harmonic potential: mu = -gamma * (x - x0, y - y0) (default)
        - Linear drift: constant velocity field
        - Custom potential: -grad(V) for potential V

    Key phenomena:
        - Relaxation to equilibrium (Gaussian for harmonic potential)
        - Balance between drift (deterministic) and diffusion (stochastic)
        - Probability conservation (integral of p remains constant)
        - Connection to Langevin equation via Ito calculus

    Reference: Fokker (1914), Planck (1917), Risken (1996)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="fokker-planck",
            category="physics",
            description="Fokker-Planck equation for probability density evolution",
            equations={
                "p": "-d_dx(mu_x * p) - d_dy(mu_y * p) + D * laplace(p)",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.1,
                    description="Diffusion coefficient",
                    min_value=0.001,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="gamma",
                    default=0.5,
                    description="Drift strength (for harmonic potential)",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="x0",
                    default=0.0,
                    description="Center x-coordinate of potential well (relative to domain center)",
                    min_value=-1.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="y0",
                    default=0.0,
                    description="Center y-coordinate of potential well (relative to domain center)",
                    min_value=-1.0,
                    max_value=1.0,
                ),
            ],
            num_fields=1,
            field_names=["p"],
            reference="Risken (1996) The Fokker-Planck Equation",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Fokker-Planck PDE.

        Args:
            parameters: Dictionary with D, gamma, x0, y0.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        D = parameters.get("D", 0.1)
        gamma = parameters.get("gamma", 0.5)
        x0_rel = parameters.get("x0", 0.0)
        y0_rel = parameters.get("y0", 0.0)

        # Get domain bounds to compute absolute center
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        x_center = (x_bounds[0] + x_bounds[1]) / 2
        y_center = (y_bounds[0] + y_bounds[1]) / 2
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[0]

        # Convert relative offset to absolute position
        x0 = x_center + x0_rel * Lx / 2
        y0 = y_center + y0_rel * Ly / 2

        # Fokker-Planck with harmonic potential drift:
        # mu = -gamma * (x - x0, y - y0)
        # dp/dt = -d/dx(mu_x * p) - d/dy(mu_y * p) + D * laplace(p)
        #       = -d/dx(-gamma * (x - x0) * p) - d/dy(-gamma * (y - y0) * p) + D * laplace(p)
        #       = gamma * d/dx((x - x0) * p) + gamma * d/dy((y - y0) * p) + D * laplace(p)
        #
        # Using product rule: d/dx((x - x0) * p) = p + (x - x0) * d_dx(p)
        # Similarly for y.
        #
        # dp/dt = gamma * (p + (x - x0) * d_dx(p)) + gamma * (p + (y - y0) * d_dy(p)) + D * laplace(p)
        #       = 2 * gamma * p + gamma * (x - x0) * d_dx(p) + gamma * (y - y0) * d_dy(p) + D * laplace(p)

        return PDE(
            rhs={
                "p": f"2 * {gamma} * p + {gamma} * (x - {x0}) * d_dx(p) + {gamma} * (y - {y0}) * d_dy(p) + {D} * laplace(p)"
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
        """Create initial state for Fokker-Planck equation.

        Default: Gaussian probability distribution offset from potential center.
        """
        if ic_type in ("fokker-planck-default", "default"):
            seed = ic_params.get("seed")
            if seed is not None:
                np.random.seed(seed)

            # Parameters for initial Gaussian
            amplitude = ic_params.get("amplitude", 1.0)
            sigma = ic_params.get("sigma", 0.15)  # Width relative to domain
            # Initial center offset (relative to domain center)
            init_x_offset = ic_params.get("init_x_offset", 0.3)
            init_y_offset = ic_params.get("init_y_offset", 0.3)

            # Get domain info
            x_bounds = grid.axes_bounds[0]
            y_bounds = grid.axes_bounds[1]
            x_center = (x_bounds[0] + x_bounds[1]) / 2
            y_center = (y_bounds[0] + y_bounds[1]) / 2
            Lx = x_bounds[1] - x_bounds[0]
            Ly = y_bounds[1] - y_bounds[0]

            # Initial Gaussian center (offset from domain center)
            x0_init = x_center + init_x_offset * Lx / 2
            y0_init = y_center + init_y_offset * Ly / 2

            # Convert sigma to absolute width
            sigma_abs = sigma * min(Lx, y_bounds[1] - y_bounds[0])

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            # Gaussian probability distribution
            data = amplitude * np.exp(
                -((X - x0_init) ** 2 + (Y - y0_init) ** 2) / (2 * sigma_abs**2)
            )

            # Normalize so integral equals 1 (probability density)
            dx = x[1] - x[0] if len(x) > 1 else 1.0
            dy = y[1] - y[0] if len(y) > 1 else 1.0
            total = np.sum(data) * dx * dy
            if total > 0:
                data = data / total

            return ScalarField(grid, data)

        if ic_type == "gaussian-ring":
            # Ring-shaped initial distribution
            seed = ic_params.get("seed")
            if seed is not None:
                np.random.seed(seed)

            radius = ic_params.get("radius", 0.3)  # Relative to domain
            width = ic_params.get("width", 0.05)  # Ring thickness

            x_bounds = grid.axes_bounds[0]
            y_bounds = grid.axes_bounds[1]
            x_center = (x_bounds[0] + x_bounds[1]) / 2
            y_center = (y_bounds[0] + y_bounds[1]) / 2
            L = min(x_bounds[1] - x_bounds[0], y_bounds[1] - y_bounds[0])

            radius_abs = radius * L / 2
            width_abs = width * L

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            r = np.sqrt((X - x_center) ** 2 + (Y - y_center) ** 2)
            data = np.exp(-((r - radius_abs) ** 2) / (2 * width_abs**2))

            # Normalize
            dx = x[1] - x[0] if len(x) > 1 else 1.0
            dy = y[1] - y[0] if len(y) > 1 else 1.0
            total = np.sum(data) * dx * dy
            if total > 0:
                data = data / total

            return ScalarField(grid, data)

        if ic_type == "double-gaussian":
            # Two separate Gaussian peaks
            seed = ic_params.get("seed")
            if seed is not None:
                np.random.seed(seed)

            sigma = ic_params.get("sigma", 0.1)
            sep = ic_params.get("separation", 0.4)  # Separation between peaks

            x_bounds = grid.axes_bounds[0]
            y_bounds = grid.axes_bounds[1]
            x_center = (x_bounds[0] + x_bounds[1]) / 2
            y_center = (y_bounds[0] + y_bounds[1]) / 2
            Lx = x_bounds[1] - x_bounds[0]

            sigma_abs = sigma * Lx
            sep_abs = sep * Lx / 2

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            # Two Gaussians offset in x
            g1 = np.exp(
                -((X - (x_center - sep_abs)) ** 2 + (Y - y_center) ** 2)
                / (2 * sigma_abs**2)
            )
            g2 = np.exp(
                -((X - (x_center + sep_abs)) ** 2 + (Y - y_center) ** 2)
                / (2 * sigma_abs**2)
            )
            data = g1 + g2

            # Normalize
            dx = x[1] - x[0] if len(x) > 1 else 1.0
            dy = y[1] - y[0] if len(y) > 1 else 1.0
            total = np.sum(data) * dx * dy
            if total > 0:
                data = data / total

            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        D = parameters.get("D", 0.1)
        gamma = parameters.get("gamma", 0.5)
        x0 = parameters.get("x0", 0.0)
        y0 = parameters.get("y0", 0.0)

        return {
            "p": f"2 * {gamma} * p + {gamma} * (x - {x0}) * d_dx(p) + {gamma} * (y - {y0}) * d_dy(p) + {D} * laplace(p)",
        }

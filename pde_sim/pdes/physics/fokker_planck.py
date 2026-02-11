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
                PDEParameter("D", "Diffusion coefficient"),
                PDEParameter("gamma", "Drift strength (for harmonic potential)"),
                PDEParameter("x0", "Center x-coordinate of potential well (relative to domain center)"),
                PDEParameter("y0", "Center y-coordinate of potential well (relative to domain center)"),
            ],
            num_fields=1,
            field_names=["p"],
            reference="Risken (1996) The Fokker-Planck Equation",
            supported_dimensions=[2],  # Currently 2D only (uses d_dy, y0 param)
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
        D = parameters["D"]
        gamma = parameters["gamma"]
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

        # Fokker-Planck with harmonic potential drift in conservative form:
        # dp/dt = d_dx(gamma*(x-x0)*p) + d_dy(gamma*(y-y0)*p) + D*laplace(p)
        #
        # The conservative form preserves probability when paired with
        # zero-flux Robin BCs: dp/dn + gamma*(x_boundary - x0)/D * p = 0.
        # This ensures J·n = gamma*(x-x0)*p + D*dp/dx = 0 at boundaries.
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds

        pde_bc = [
            [
                {"type": "mixed", "value": -gamma * (x_min - x0) / D, "const": 0},
                {"type": "mixed", "value": gamma * (x_max - x0) / D, "const": 0},
            ],
            [
                {"type": "mixed", "value": -gamma * (y_min - y0) / D, "const": 0},
                {"type": "mixed", "value": gamma * (y_max - y0) / D, "const": 0},
            ],
        ]

        return PDE(
            rhs={
                "p": (
                    f"d_dx({gamma} * (x - {x0}) * p)"
                    f" + d_dy({gamma} * (y - {y0}) * p)"
                    f" + {D} * laplace(p)"
                )
            },
            bc=pde_bc,
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
            init_x_offset = ic_params.get("init_x_offset")
            init_y_offset = ic_params.get("init_y_offset")
            if (
                init_x_offset is None or init_x_offset == "random"
                or init_y_offset is None or init_y_offset == "random"
            ):
                raise ValueError("fokker-planck default requires init_x_offset and init_y_offset (or random)")

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

    def get_position_params(self, ic_type: str) -> set[str]:
        if ic_type in ("fokker-planck-default", "default"):
            return {"init_x_offset", "init_y_offset"}
        return super().get_position_params(ic_type)

    def resolve_ic_params(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> dict[str, Any]:
        if ic_type in ("fokker-planck-default", "default"):
            resolved = ic_params.copy()
            required = ["init_x_offset", "init_y_offset"]
            for key in required:
                if key not in resolved:
                    raise ValueError("fokker-planck default requires init_x_offset and init_y_offset (or random)")
            if any(resolved[key] == "random" for key in required):
                rng = np.random.default_rng(resolved.get("seed"))
                if resolved["init_x_offset"] == "random":
                    resolved["init_x_offset"] = rng.uniform(-0.5, 0.5)
                if resolved["init_y_offset"] == "random":
                    resolved["init_y_offset"] = rng.uniform(-0.5, 0.5)
            if any(resolved[key] is None or resolved[key] == "random" for key in required):
                raise ValueError("fokker-planck default requires init_x_offset and init_y_offset (or random)")
            return resolved

        return super().resolve_ic_params(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        D = parameters["D"]
        gamma = parameters["gamma"]
        x0 = parameters.get("x0", 0.0)
        y0 = parameters.get("y0", 0.0)

        return {
            "p": f"d_dx({gamma} * (x - {x0}) * p) + d_dy({gamma} * (y - {y0}) * p) + {D} * laplace(p)",
        }

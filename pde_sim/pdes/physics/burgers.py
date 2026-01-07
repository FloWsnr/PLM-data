"""Burgers' equation for shock wave formation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("burgers")
class BurgersPDE(ScalarPDEPreset):
    """Burgers' equation (viscous).

    Based on visualpde.com formulation:

        du/dt = -u * d_dx(u) + epsilon * laplace(u)

    The simplest nonlinear PDE combining advection and diffusion.

    Key phenomena:
        - Wave steepening: larger amplitudes travel faster
        - Shock formation: in inviscid limit, discontinuities form
        - Shock thickness: epsilon determines shock width
        - Shock interaction: multiple shocks can merge

    The viscous Burgers equation is exactly solvable via Cole-Hopf transformation.

    Reference: Burgers (1948), Cole (1951), Hopf (1950)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="burgers",
            category="physics",
            description="Burgers' equation for shock wave formation",
            equations={
                "u": "-u * d_dx(u) + epsilon * laplace(u)",
            },
            parameters=[
                PDEParameter(
                    name="epsilon",
                    default=0.05,
                    description="Viscosity coefficient",
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Burgers (1948) Mathematical model for turbulence",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Burgers' equation PDE.

        Args:
            parameters: Dictionary with epsilon.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        epsilon = parameters.get("epsilon", 0.05)

        # Viscous Burgers' equation:
        # du/dt = -u * d_dx(u) + epsilon * laplace(u)
        return PDE(
            rhs={"u": f"-u * d_dx(u) + {epsilon} * laplace(u)"},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> ScalarField:
        """Create initial state for Burgers' equation.

        Default: Gaussian pulse for shock formation demonstration.
        """
        if ic_type in ("burgers-default", "default"):
            # Gaussian pulse at x = L/5
            x_bounds = grid.axes_bounds[0]
            Lx = x_bounds[1] - x_bounds[0]
            x0 = x_bounds[0] + Lx / 5  # Pulse position

            amplitude = ic_params.get("amplitude", 1.0)
            width = ic_params.get("width", 0.1) * Lx
            seed = ic_params.get("seed")
            if seed is not None:
                np.random.seed(seed)

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            if len(grid.shape) > 1:
                y_bounds = grid.axes_bounds[1]
                y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
                X, Y = np.meshgrid(x, y, indexing="ij")
                # 2D: Gaussian depends only on x
                data = amplitude * np.exp(-((X - x0) ** 2) / (2 * width**2))
            else:
                data = amplitude * np.exp(-((x - x0) ** 2) / (2 * width**2))

            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        epsilon = parameters.get("epsilon", 0.05)

        return {
            "u": f"-u * d_dx(u) + {epsilon} * laplace(u)",
        }

"""Korteweg-de Vries (KdV) equation for solitons."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import PDEMetadata, PDEParameter, ScalarPDEPreset
from .. import register_pde


@register_pde("korteweg-de-vries")
class KortewegDeVriesPDE(ScalarPDEPreset):
    """Korteweg-de Vries (KdV) equation for soliton dynamics.

    Based on visualpde.com formulation:

        dphi/dt = -d_dx(d_dx(d_dx(phi))) - 6 * phi * d_dx(phi)

    The prototypical integrable PDE describing soliton dynamics.
    A precise balance between nonlinear steepening and dispersion
    produces stable, particle-like traveling waves.

    Key properties:
        - Complete integrability via Inverse Scattering Transform
        - Infinite conservation laws
        - Soliton solutions that maintain shape
        - Elastic collisions: solitons pass through each other with only phase shifts

    Single soliton solution:
        phi(x,t) = (c/2) * sech^2(sqrt(c)/2 * (x - c*t - x0))

    where c is the speed (taller solitons move faster).

    Reference: Korteweg & de Vries (1895), Zabusky & Kruskal (1965)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="korteweg-de-vries",
            category="physics",
            description="KdV equation for soliton dynamics",
            equations={
                "phi": "-d_dx(d_dx(d_dx(phi))) - 6 * phi * d_dx(phi)",
            },
            parameters=[
                # The standard KdV form has no adjustable parameters
                # but we include a scaling for flexibility
                PDEParameter(
                    name="scale",
                    default=1.0,
                    description="Overall scaling factor",
                    min_value=0.1,
                    max_value=10.0,
                ),
            ],
            num_fields=1,
            field_names=["phi"],
            reference="Korteweg & de Vries (1895)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the KdV equation PDE.

        Args:
            parameters: Dictionary with optional scale.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        scale = parameters.get("scale", 1.0)

        # Standard KdV equation:
        # dphi/dt = -phi_xxx - 6*phi*phi_x
        # Note: d_dx(d_dx(d_dx(phi))) is the third derivative
        return PDE(
            rhs={"phi": f"-{scale} * (d_dx(d_dx(d_dx(phi))) + 6 * phi * d_dx(phi))"},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> ScalarField:
        """Create initial state - typically two solitons for collision demo.

        Default: Two solitons of different amplitudes for collision demonstration.
        """
        if ic_type in ("korteweg-de-vries-default", "default", "two-solitons"):
            # Two-soliton initial condition
            x_bounds = grid.axes_bounds[0]
            Lx = x_bounds[1] - x_bounds[0]

            # Soliton 1: amplitude 2, at x = L/4
            A1 = ic_params.get("A1", 2.0)
            x1 = x_bounds[0] + Lx / 4

            # Soliton 2: amplitude ~1.125 (0.75^2 * 2), at x = 0.4*L
            A2 = ic_params.get("A2", 0.75**2 * 2)
            x2 = x_bounds[0] + 0.4 * Lx

            seed = ic_params.get("seed")
            if seed is not None:
                np.random.seed(seed)

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            if len(grid.shape) > 1:
                y_bounds = grid.axes_bounds[1]
                y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
                X, Y = np.meshgrid(x, y, indexing="ij")
                # 1D solitons in 2D domain (constant in y)
                # sech^2 profile: phi = (c/2) * sech^2(sqrt(c)/2 * (x - x0))
                # For amplitude A, c = 2*A
                c1 = 2 * A1
                c2 = 2 * A2
                soliton1 = (c1 / 2) / np.cosh(np.sqrt(c1) / 2 * (X - x1)) ** 2
                soliton2 = (c2 / 2) / np.cosh(np.sqrt(c2) / 2 * (X - x2)) ** 2
                data = soliton1 + soliton2
            else:
                c1 = 2 * A1
                c2 = 2 * A2
                soliton1 = (c1 / 2) / np.cosh(np.sqrt(c1) / 2 * (x - x1)) ** 2
                soliton2 = (c2 / 2) / np.cosh(np.sqrt(c2) / 2 * (x - x2)) ** 2
                data = soliton1 + soliton2

            return ScalarField(grid, data)

        if ic_type == "single-soliton":
            # Single soliton
            x_bounds = grid.axes_bounds[0]
            Lx = x_bounds[1] - x_bounds[0]

            A = ic_params.get("amplitude", 1.0)
            x0 = ic_params.get("x0", x_bounds[0] + Lx / 4)

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            if len(grid.shape) > 1:
                y_bounds = grid.axes_bounds[1]
                y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
                X, Y = np.meshgrid(x, y, indexing="ij")
                c = 2 * A
                data = (c / 2) / np.cosh(np.sqrt(c) / 2 * (X - x0)) ** 2
            else:
                c = 2 * A
                data = (c / 2) / np.cosh(np.sqrt(c) / 2 * (x - x0)) ** 2

            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        return {
            "phi": "-d_dx(d_dx(d_dx(phi))) - 6 * phi * d_dx(phi)",
        }

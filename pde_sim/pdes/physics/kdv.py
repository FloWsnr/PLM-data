"""Korteweg-de Vries (KdV) equation - classical soliton equation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import PDEMetadata, PDEParameter, ScalarPDEPreset
from .. import register_pde


@register_pde("kdv")
class KdVPDE(ScalarPDEPreset):
    """Korteweg-de Vries (KdV) equation - the canonical soliton equation.

    The KdV equation:

        du/dt = -d³u/dx³ - 6*u*du/dx - b*∇⁴u

    This is the foundational equation of soliton theory, describing weakly
    nonlinear shallow water waves, ion-acoustic waves in plasmas, and
    many other dispersive wave phenomena.

    Key properties:
        - Completely integrable (infinite conservation laws)
        - Exact N-soliton solutions via inverse scattering
        - Balance of dispersion (u_xxx) and nonlinearity (u*u_x)
        - Soliton amplitude determines speed: taller = faster

    The implementation uses a pseudo-2D approach where the KdV dynamics
    evolve along the x-axis, with periodic boundaries in y to allow
    2D visualization. An optional biharmonic dissipation term helps
    reduce numerical artifacts.

    Initial conditions:
        - Single soliton: sech²(k(x-x0)) with amplitude 2k²
        - Two solitons: superposition for collision dynamics
        - N-wave: initial bump that evolves into multiple solitons

    Reference: Korteweg & de Vries (1895), Gardner et al. (1967)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="kdv",
            category="physics",
            description="Korteweg-de Vries equation for soliton dynamics",
            equations={
                "u": "-d_dxdxdx(u) - 6 * u * d_dx(u) - b * laplace(laplace(u))",
            },
            parameters=[
                PDEParameter(
                    name="b",
                    default=0.0001,
                    description="Biharmonic dissipation coefficient (stabilization)",
                    min_value=0.0,
                    max_value=0.1,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Korteweg & de Vries (1895), Gardner et al. (1967)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the KdV equation PDE.

        Args:
            parameters: Dictionary with dissipation coefficient b.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        b = parameters.get("b", 0.0001)

        # KdV equation with optional biharmonic dissipation:
        # du/dt = -d³u/dx³ - 6*u*du/dx - b*∇⁴u
        #
        # Using py-pde's d_dx operator for derivatives.
        # The third derivative d³u/dx³ = d_dx(d_dx(d_dx(u)))
        rhs = f"-d_dx(d_dx(d_dx(u))) - 6 * u * d_dx(u) - {b} * laplace(laplace(u))"

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
        """Create initial state for KdV equation.

        Supported initial conditions:
            - default/soliton: Single sech² soliton
            - two-solitons: Two solitons for collision demonstration
            - n-wave: Initial bump that breaks into solitons
            - random-solitons: Multiple randomly placed solitons
        """
        x_bounds = grid.axes_bounds[0]
        Lx = x_bounds[1] - x_bounds[0]

        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])

        # Handle 2D grid
        if len(grid.shape) > 1:
            y_bounds = grid.axes_bounds[1]
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")
            x_coord = X  # Use X for 2D
        else:
            x_coord = x

        if ic_type in ("kdv-default", "default", "soliton"):
            # Single soliton: u = 2k² sech²(k(x-x0))
            # Speed c = 4k², so amplitude = c/2
            k = ic_params.get("k", 0.5)  # Width parameter
            x0 = ic_params.get("x0", x_bounds[0] + Lx * 0.25)  # Start at left quarter

            amplitude = 2 * k**2
            data = amplitude / np.cosh(k * (x_coord - x0)) ** 2

            return ScalarField(grid, data)

        if ic_type == "two-solitons":
            # Two solitons for collision demonstration
            # Taller soliton (larger k) moves faster and will overtake
            k1 = ic_params.get("k1", 0.6)  # Taller, faster soliton
            k2 = ic_params.get("k2", 0.4)  # Shorter, slower soliton
            x1 = ic_params.get("x1", x_bounds[0] + Lx * 0.15)  # Behind
            x2 = ic_params.get("x2", x_bounds[0] + Lx * 0.35)  # Ahead

            amp1 = 2 * k1**2
            amp2 = 2 * k2**2

            soliton1 = amp1 / np.cosh(k1 * (x_coord - x1)) ** 2
            soliton2 = amp2 / np.cosh(k2 * (x_coord - x2)) ** 2

            data = soliton1 + soliton2

            return ScalarField(grid, data)

        if ic_type == "n-wave":
            # Initial bump that evolves into multiple solitons
            # A smooth initial pulse breaks into a train of solitons
            amplitude = ic_params.get("amplitude", 1.0)
            width = ic_params.get("width", 0.1) * Lx
            x0 = ic_params.get("x0", x_bounds[0] + Lx * 0.2)

            # Gaussian bump
            data = amplitude * np.exp(-((x_coord - x0) ** 2) / (2 * width**2))

            return ScalarField(grid, data)

        if ic_type == "offset-soliton":
            # Soliton offset from center for observing propagation
            k = ic_params.get("k", 0.5)
            x0 = ic_params.get("x0", x_bounds[0] + Lx * 0.2)

            amplitude = 2 * k**2
            data = amplitude / np.cosh(k * (x_coord - x0)) ** 2

            return ScalarField(grid, data)

        if ic_type == "random-solitons":
            # N randomly placed solitons with random widths
            n = ic_params.get("n", 3)
            k_min = ic_params.get("k_min", 0.3)
            k_max = ic_params.get("k_max", 0.7)
            margin = ic_params.get("margin", 0.1)

            seed = kwargs.get("seed", None)
            rng = np.random.default_rng(seed)

            x_range = (x_bounds[0] + margin * Lx, x_bounds[0] + (1 - margin) * Lx * 0.5)

            if len(grid.shape) > 1:
                data = np.zeros_like(X)
            else:
                data = np.zeros_like(x)

            for _ in range(n):
                xi = rng.uniform(x_range[0], x_range[1])
                ki = rng.uniform(k_min, k_max)
                ampi = 2 * ki**2

                data += ampi / np.cosh(ki * (x_coord - xi)) ** 2

            return ScalarField(grid, data)

        # Fall back to generic initial condition
        return create_initial_condition(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        b = parameters.get("b", 0.0001)

        if b > 0:
            return {
                "u": f"-d_dxdxdx(u) - 6 * u * d_dx(u) - {b} * laplace(laplace(u))",
            }
        else:
            return {
                "u": "-d_dxdxdx(u) - 6 * u * d_dx(u)",
            }

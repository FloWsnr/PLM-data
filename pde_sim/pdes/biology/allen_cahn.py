"""Allen-Cahn (bistable) equations."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("allen-cahn")
class AllenCahnPDE(ScalarPDEPreset):
    """Allen-Cahn (bistable) equation.

    The Allen-Cahn equation describes phase transitions:

        du/dt = D * laplace(u) + u * (1 - u²)

    or with threshold parameter a:

        du/dt = D * laplace(u) + u * (u - a) * (1 - u)

    Exhibits bistable traveling fronts.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="allen-cahn",
            category="biology",
            description="Allen-Cahn bistable equation",
            equations={
                "u": "D * laplace(u) + u * (u - a) * (1 - u)",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.1,
                    description="Diffusion coefficient",
                    min_value=0.01,
                    max_value=0.5,
                ),
                PDEParameter(
                    name="a",
                    default=0.5,
                    description="Threshold parameter (0 < a < 1)",
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Bistable traveling fronts, Allee effect",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        a = parameters.get("a", 0.5)

        return PDE(
            rhs={"u": f"{D} * laplace(u) + u * (u - {a}) * (1 - u)"},
            bc=self._convert_bc(bc),
        )


@register_pde("allen-cahn-standard")
class StandardAllenCahnPDE(ScalarPDEPreset):
    """Standard Allen-Cahn equation with symmetric double-well potential.

    The standard Allen-Cahn equation from the py-pde library:

        dc/dt = gamma * laplace(c) - c³ + c

    or equivalently:

        dc/dt = gamma * laplace(c) + c * (1 - c²)

    where:
        - c is the order parameter (phase field)
        - gamma is the interfacial width parameter (diffusion)
        - The reaction term derives from double-well potential F(c) = (1/4)(c² - 1)²

    Equilibria: c = -1 (stable), c = 0 (unstable), c = +1 (stable)

    This is the symmetric form with stable phases at ±1.
    See 'allen-cahn' for the asymmetric bistable (Nagumo) form.

    Reference: Allen & Cahn (1979)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="allen-cahn-standard",
            category="biology",
            description="Standard Allen-Cahn with symmetric double-well",
            equations={
                "c": "gamma * laplace(c) - c**3 + c",
            },
            parameters=[
                PDEParameter(
                    name="gamma",
                    default=0.1,
                    description="Interfacial width parameter",
                    min_value=0.01,
                    max_value=0.5,
                ),
                PDEParameter(
                    name="mobility",
                    default=1.0,
                    description="Mobility (rate of evolution)",
                    min_value=0.1,
                    max_value=2.0,
                ),
            ],
            num_fields=1,
            field_names=["c"],
            reference="Phase field with symmetric double-well potential",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the standard Allen-Cahn PDE.

        Args:
            parameters: Dictionary containing 'gamma' and 'mobility'.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        gamma = parameters.get("gamma", 1.0)
        mobility = parameters.get("mobility", 1.0)

        # dc/dt = mobility * (gamma * laplace(c) - c³ + c)
        if mobility == 1.0:
            rhs = f"{gamma} * laplace(c) - c**3 + c"
        else:
            rhs = f"{mobility} * ({gamma} * laplace(c) - c**3 + c)"

        return PDE(
            rhs={"c": rhs},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial state for standard Allen-Cahn.

        Common initial conditions:
        - random: Small random noise to trigger spinodal decomposition
        - step: Sharp interface between phases
        - tanh: Pre-formed interface profile
        """
        seed = ic_params.get("seed")
        if seed is not None:
            np.random.seed(seed)

        if ic_type in ("allen-cahn-standard-default", "default", "random"):
            # Small random perturbations for spinodal decomposition
            amplitude = ic_params.get("amplitude", 0.1)
            data = amplitude * np.random.randn(*grid.shape)
            return ScalarField(grid, data)

        if ic_type == "step":
            # Step function: -1 on left, +1 on right
            x_bounds = grid.axes_bounds[0]
            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(grid.axes_bounds[1][0], grid.axes_bounds[1][1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            mid = (x_bounds[0] + x_bounds[1]) / 2
            data = np.where(X < mid, -1.0, 1.0)
            return ScalarField(grid, data)

        if ic_type == "tanh":
            # Tanh profile (exact interface solution)
            gamma = ic_params.get("gamma", 1.0)
            x_bounds = grid.axes_bounds[0]
            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(grid.axes_bounds[1][0], grid.axes_bounds[1][1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            mid = (x_bounds[0] + x_bounds[1]) / 2
            width = np.sqrt(2 * gamma)
            data = np.tanh((X - mid) / width)
            return ScalarField(grid, data)

        # Default fallback
        data = 0.1 * np.random.randn(*grid.shape)
        return ScalarField(grid, data)

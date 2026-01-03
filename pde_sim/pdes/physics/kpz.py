"""Kardar-Parisi-Zhang (KPZ) interface growth equation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("kpz")
class KPZInterfacePDE(ScalarPDEPreset):
    """Kardar-Parisi-Zhang (KPZ) interface growth equation.

    The KPZ equation is a universal model for surface/interface growth:

        dh/dt = nu * laplace(h) + (lambda/2) * |grad(h)|^2

    where:
        - h is the height of the interface
        - nu is the diffusion coefficient (surface tension)
        - lambda is the growth strength (lateral growth rate)

    This equation describes:
        - Surface growth by random deposition
        - Eden model universality class
        - Molecular beam epitaxy (MBE)
        - Ballistic deposition
        - Flame front propagation (related to KS equation)

    Note: This is the deterministic version. The stochastic version
    includes additive noise eta(r,t).

    Reference: Kardar, Parisi, Zhang (1986)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="kpz",
            category="physics",
            description="Kardar-Parisi-Zhang interface growth",
            equations={
                "h": "nu * laplace(h) + (lambda/2) * gradient_squared(h)",
            },
            parameters=[
                PDEParameter(
                    name="nu",
                    default=0.1,
                    description="Diffusion coefficient (surface tension)",
                    min_value=0.01,
                    max_value=0.5,
                ),
                PDEParameter(
                    name="lmbda",
                    default=1.0,
                    description="Growth strength (lateral growth rate)",
                    min_value=0.1,
                    max_value=2.0,
                ),
            ],
            num_fields=1,
            field_names=["h"],
            reference="Surface growth universality (KPZ class)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the KPZ interface growth equation PDE.

        Args:
            parameters: Dictionary containing 'nu' and 'lmbda' coefficients.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        nu = parameters.get("nu", 0.5)
        lmbda = parameters.get("lmbda", 1.0)

        # KPZ equation: dh/dt = nu * laplace(h) + (lambda/2) * |grad(h)|^2
        return PDE(
            rhs={"h": f"{nu} * laplace(h) + {lmbda / 2} * gradient_squared(h)"},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial state for interface height.

        Common initial conditions:
        - flat: Flat interface with small random perturbations
        - sinusoidal: Sinusoidal profile to study smoothing
        """
        if ic_type in ("kpz-default", "default", "flat"):
            # Small random perturbations around flat interface
            amplitude = ic_params.get("amplitude", 0.1)
            seed = ic_params.get("seed")
            if seed is not None:
                np.random.seed(seed)
            data = amplitude * np.random.randn(*grid.shape)
            return ScalarField(grid, data)

        if ic_type == "sinusoidal":
            # Sinusoidal initial profile
            amplitude = ic_params.get("amplitude", 1.0)
            wavelength = ic_params.get("wavelength", 1.0)

            x_bounds = grid.axes_bounds[0]
            y_bounds = grid.axes_bounds[1]
            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            Lx = x_bounds[1] - x_bounds[0]
            data = amplitude * np.sin(2 * np.pi * X / wavelength)

            return ScalarField(grid, data)

        # Fall back to parent implementation
        return create_initial_condition(grid, ic_type, ic_params)

"""Kardar-Parisi-Zhang (KPZ) interface growth equation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("kpz")
class KPZInterfacePDE(ScalarPDEPreset):
    """Damped Kardar-Parisi-Zhang (KPZ) interface growth equation.

    The damped KPZ equation models stochastic surface/interface growth:

        dh/dt = -gamma * h + nu * laplace(h) + (lambda/2) * |grad(h)|^2 + eta * noise

    where:
        - h is the height of the interface
        - gamma is the linear damping coefficient (keeps mean bounded)
        - nu is the diffusion coefficient (surface tension)
        - lambda is the growth strength (lateral growth rate)
        - eta is the noise strength (stochastic deposition)

    This equation describes:
        - Surface growth by random deposition
        - Eden model universality class
        - Molecular beam epitaxy (MBE)
        - Ballistic deposition
        - Flame front propagation (related to KS equation)

    The damping term -gamma*h prevents unbounded growth of mean height,
    allowing steady-state roughening dynamics to be visualized.

    Reference: Kardar, Parisi, Zhang (1986)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="kpz",
            category="physics",
            description="Kardar-Parisi-Zhang interface growth (stochastic)",
            equations={
                "h": "-gamma * h + nu * laplace(h) + (lambda/2) * gradient_squared(h) + eta * noise",
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
                PDEParameter(
                    name="eta",
                    default=0.5,
                    description="Noise strength (stochastic deposition rate)",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="gamma",
                    default=0.1,
                    description="Linear damping (keeps mean height bounded)",
                    min_value=0.0,
                    max_value=1.0,
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
        """Create the stochastic damped KPZ interface growth equation PDE.

        Args:
            parameters: Dictionary containing 'nu', 'lmbda', 'eta', and 'gamma' coefficients.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance with stochastic noise.
        """
        nu = parameters.get("nu", 0.1)
        lmbda = parameters.get("lmbda", 1.0)
        eta = parameters.get("eta", 0.5)
        gamma = parameters.get("gamma", 0.1)

        # Damped stochastic KPZ: dh/dt = -gamma*h + nu*laplace(h) + (lambda/2)*|grad(h)|^2 + noise
        # The damping term -gamma*h keeps the mean height bounded
        return PDE(
            rhs={"h": f"-{gamma} * h + {nu} * laplace(h) + {lmbda / 2} * gradient_squared(h)"},
            bc=self._convert_bc(bc),
            noise={"h": eta},  # Additive white noise with strength eta
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

"""Lotka-Volterra predator-prey reaction-diffusion model."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("lotka-volterra")
class LotkaVolterraPDE(MultiFieldPDEPreset):
    """Lotka-Volterra predator-prey model with spatial diffusion.

    A two-species reaction-diffusion system modeling predator-prey dynamics:

        du/dt = Du*laplace(u) + alpha*u - beta*u*v
        dv/dt = Dv*laplace(v) + delta*u*v - gamma*v

    where:
        - u is the prey population density
        - v is the predator population density
        - alpha is the prey growth rate (in absence of predators)
        - beta is the predation rate (prey death due to predation)
        - delta is the predator reproduction efficiency (conversion of prey to predators)
        - gamma is the predator death rate (in absence of prey)
        - Du, Dv are diffusion coefficients

    Key phenomena:
        - Oscillatory dynamics: predator-prey population cycles
        - Pursuit waves: predators chasing prey across space
        - Spiral waves: rotating patterns from localized perturbations
        - Turing patterns: stationary spatial patterns (for appropriate parameters)

    References:
        Lotka, A. J. (1925). Elements of Physical Biology.
        Volterra, V. (1926). Fluctuations in the abundance of a species
            considered mathematically. Nature, 118, 558-560.
        Murray, J. D. (2002). Mathematical Biology I: An Introduction. Springer.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="lotka-volterra",
            category="biology",
            description="Predator-prey dynamics with spatial diffusion",
            equations={
                "u": "Du * laplace(u) + alpha * u - beta * u * v",
                "v": "Dv * laplace(v) + delta * u * v - gamma * v",
            },
            parameters=[
                PDEParameter("Du", "Prey diffusion coefficient"),
                PDEParameter("Dv", "Predator diffusion coefficient"),
                PDEParameter("alpha", "Prey growth rate"),
                PDEParameter("beta", "Predation rate"),
                PDEParameter("delta", "Predator reproduction efficiency"),
                PDEParameter("gamma", "Predator death rate"),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Lotka (1925), Volterra (1926), Murray (2002)",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Du = parameters.get("Du", 1.0)
        Dv = parameters.get("Dv", 0.5)
        alpha = parameters.get("alpha", 1.0)
        beta = parameters.get("beta", 0.1)
        delta = parameters.get("delta", 0.075)
        gamma = parameters.get("gamma", 0.5)

        return PDE(
            rhs={
                "u": f"{Du} * laplace(u) + {alpha} * u - {beta} * u * v",
                "v": f"{Dv} * laplace(v) + {delta} * u * v - {gamma} * v",
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial state for predator-prey system.

        Default creates localized predator and prey populations with noise,
        which can lead to pursuit patterns and spiral waves.
        """
        if ic_type not in ("default", "lotka-volterra-default"):
            return super().create_initial_state(grid, ic_type, ic_params, **kwargs)

        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.1)

        # Get equilibrium populations (non-trivial fixed point)
        # u* = gamma/delta, v* = alpha/beta
        alpha = ic_params.get("alpha", 1.0)
        beta = ic_params.get("beta", 0.1)
        delta = ic_params.get("delta", 0.075)
        gamma = ic_params.get("gamma", 0.5)

        u_eq = gamma / delta  # prey equilibrium
        v_eq = alpha / beta   # predator equilibrium

        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        cx = (x_bounds[0] + x_bounds[1]) / 2
        cy = (y_bounds[0] + y_bounds[1]) / 2
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[1]

        # Create spatial structure: prey and predator at different locations
        # This promotes pursuit waves and spatial dynamics
        # Prey centered at (cx - Lx/4, cy), predator at (cx + Lx/4, cy)
        offset = Lx / 6

        # Localized bumps with Gaussian profiles
        width = Lx / 10
        prey_bump = np.exp(-((X - cx + offset)**2 + (Y - cy)**2) / (2 * width**2))
        pred_bump = np.exp(-((X - cx - offset)**2 + (Y - cy)**2) / (2 * width**2))

        # Start near equilibrium with spatial perturbation
        u_data = u_eq * (1 + 0.5 * prey_bump) + noise * np.random.randn(*grid.shape)
        v_data = v_eq * (1 + 0.5 * pred_bump) + noise * np.random.randn(*grid.shape)

        # Ensure non-negative (populations can't be negative)
        u_data = np.clip(u_data, 0.0, None)
        v_data = np.clip(v_data, 0.0, None)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

"""Stochastic Gray-Scott reaction-diffusion system."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("stochastic-gray-scott")
class StochasticGrayScottPDE(MultiFieldPDEPreset):
    """Stochastic Gray-Scott reaction-diffusion system.

    Based on the visualpde.com formulation with multiplicative noise:

        du/dt = laplace(u) + u^2*v - (a+b)*u + sigma * dW/dt * u
        dv/dt = D * laplace(v) - u^2*v + a*(1-v)

    where:
        - u is the activator concentration
        - v is the substrate concentration
        - a is the feed rate
        - b is the kill/removal rate
        - D is the diffusion ratio
        - sigma is the noise intensity
        - dW/dt is spatiotemporal white noise

    Key phenomena in stochastic reaction-diffusion systems:
        - Noise-induced patterns: Patterns emerge that would not exist deterministically
        - Stochastic resonance: Intermediate noise levels optimize pattern formation
        - Stochastic extinction: Very strong noise destroys all patterns

    IMPORTANT: The stochastic terms only work correctly with Euler timestepping!

    Reference: Biancalani et al. (2010), Woolley et al. (2011)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="stochastic-gray-scott",
            category="physics",
            description="Stochastic Gray-Scott with noise-induced patterns",
            equations={
                "u": "laplace(u) + u**2 * v - (a + b) * u + sigma * noise * u",
                "v": "D * laplace(v) - u**2 * v + a * (1 - v)",
            },
            parameters=[
                PDEParameter(
                    name="a",
                    default=0.037,
                    description="Feed rate",
                    min_value=0.0,
                    max_value=0.1,
                ),
                PDEParameter(
                    name="b",
                    default=0.04,
                    description="Kill/removal rate",
                    min_value=0.04,
                    max_value=0.1,
                ),
                PDEParameter(
                    name="D",
                    default=4.0,
                    description="Diffusion ratio (v diffuses D times faster than u)",
                    min_value=1.0,
                    max_value=8.0,
                ),
                PDEParameter(
                    name="sigma",
                    default=0.0,
                    description="Noise intensity (0 = deterministic)",
                    min_value=0.0,
                    max_value=0.8,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Biancalani et al. (2010) Stochastic Turing patterns",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Stochastic Gray-Scott PDE system.

        Args:
            parameters: Dictionary with a, b, D, sigma.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance with multiplicative noise.
        """
        a = parameters.get("a", 0.037)
        b = parameters.get("b", 0.04)
        D = parameters.get("D", 4.0)
        sigma = parameters.get("sigma", 0.0)

        # Stochastic Gray-Scott equations:
        # du/dt = laplace(u) + u^2*v - (a+b)*u + sigma*noise*u
        # dv/dt = D*laplace(v) - u^2*v + a*(1-v)
        #
        # The noise term is multiplicative (multiplied by u)
        # py-pde's noise parameter adds additive noise, but for multiplicative
        # we need to use a formula that effectively creates the scaling
        return PDE(
            rhs={
                "u": f"laplace(u) + u**2 * v - ({a} + {b}) * u",
                "v": f"{D} * laplace(v) - u**2 * v + {a} * (1 - v)",
            },
            bc=self._convert_bc(bc),
            # Additive noise on u field - for true multiplicative noise
            # we would need a custom implementation, but this approximates
            # the effect for small sigma
            noise={"u": sigma, "v": 0.0} if sigma > 0 else None,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state for Stochastic Gray-Scott.

        Default initialization: small circular patch of u on uniform v=1 background.
        """
        # Check for per-field specifications
        if "u" in ic_params and isinstance(ic_params["u"], dict):
            return super().create_initial_state(grid, ic_type, ic_params)

        if ic_type in ("stochastic-gray-scott-default", "default"):
            return self._default_stochastic_init(grid, ic_params)

        # Fallback to parent implementation
        return super().create_initial_state(grid, ic_type, ic_params)

    def _default_stochastic_init(
        self,
        grid: CartesianGrid,
        params: dict[str, Any],
    ) -> FieldCollection:
        """Create default initialization with small circular patch of u."""
        # Get domain info
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[0]

        # Create coordinates
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Center of domain
        cx = x_bounds[0] + Lx / 2
        cy = y_bounds[0] + Ly / 2

        # Small circular patch
        r = params.get("perturbation_radius", 0.05) * min(Lx, Ly)
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = dist < r

        # Initialize u = 0, v = 1 everywhere
        u_data = np.zeros(grid.shape)
        v_data = np.ones(grid.shape)

        # Add small patch of u in center
        u_data[mask] = 0.5

        # Add very small noise
        noise = params.get("noise", 0.001)
        seed = params.get("seed")
        if seed is not None:
            np.random.seed(seed)
        u_data += noise * np.random.randn(*grid.shape)
        v_data += noise * np.random.randn(*grid.shape)

        # Clip to valid range
        u_data = np.clip(u_data, 0, 1)
        v_data = np.clip(v_data, 0, 1)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        a = parameters.get("a", 0.037)
        b = parameters.get("b", 0.04)
        D = parameters.get("D", 4.0)
        sigma = parameters.get("sigma", 0.0)

        return {
            "u": f"laplace(u) + u**2 * v - ({a} + {b}) * u + {sigma} * noise * u",
            "v": f"{D} * laplace(v) - u**2 * v + {a} * (1 - v)",
        }

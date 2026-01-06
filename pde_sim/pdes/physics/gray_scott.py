"""Gray-Scott reaction-diffusion system."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("gray-scott")
class GrayScottPDE(MultiFieldPDEPreset):
    """Gray-Scott reaction-diffusion system.

    Based on the visualpde.com formulation:

        du/dt = laplace(u) + u^2*v - (a+b)*u
        dv/dt = D * laplace(v) - u^2*v + a*(1-v)

    where:
        - u is the activator concentration (autocatalytic species)
        - v is the substrate concentration (fuel species)
        - a is the feed rate
        - b is the kill/removal rate
        - D is the diffusion ratio (v diffuses D times faster than u)

    The system exhibits an extraordinary range of patterns depending on (a,b):
        - Labyrinthine/stripe patterns
        - Stationary and pulsating spots
        - Self-replicating spots
        - Worm-like structures
        - Holes, spatiotemporal chaos
        - Moving spots (gliders)

    Reference: Pearson (1993) "Complex Patterns in a Simple System"
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="gray-scott",
            category="physics",
            description="Gray-Scott reaction-diffusion pattern formation",
            equations={
                "u": "laplace(u) + u**2 * v - (a + b) * u",
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
                    default=0.06,
                    description="Kill/removal rate",
                    min_value=0.04,
                    max_value=0.1,
                ),
                PDEParameter(
                    name="D",
                    default=2.0,
                    description="Diffusion ratio (v diffuses D times faster than u)",
                    min_value=1.0,
                    max_value=4.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Pearson (1993) Complex patterns in a simple system",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Gray-Scott PDE system.

        Args:
            parameters: Dictionary with a, b, D.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        a = parameters.get("a", 0.037)
        b = parameters.get("b", 0.06)
        D = parameters.get("D", 2.0)

        # Gray-Scott equations from reference:
        # du/dt = laplace(u) + u^2*v - (a+b)*u
        # dv/dt = D*laplace(v) - u^2*v + a*(1-v)
        return PDE(
            rhs={
                "u": f"laplace(u) + u**2 * v - ({a} + {b}) * u",
                "v": f"{D} * laplace(v) - u**2 * v + {a} * (1 - v)",
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state for Gray-Scott.

        The default initialization for Gray-Scott is:
        - u = 0 everywhere with a perturbation region where u is higher
        - v = 1 everywhere (uniform substrate)

        This creates the "seed" for pattern formation.
        """
        # Check for per-field specifications
        if "u" in ic_params and isinstance(ic_params["u"], dict):
            # Use specified per-field ICs
            return super().create_initial_state(grid, ic_type, ic_params)

        # Default Gray-Scott initialization
        if ic_type in ("gray-scott-default", "default"):
            return self._default_gray_scott_init(grid, ic_params)

        # For other IC types, create uniform backgrounds and add perturbation
        if ic_type in ("random-uniform", "random-gaussian"):
            # Create random perturbation for u
            u_field = create_initial_condition(grid, ic_type, ic_params)
            # Scale u to be small perturbations
            u_data = u_field.data * 0.1

            # v starts at 1.0
            v_data = np.ones(grid.shape)

            u = ScalarField(grid, u_data)
            u.label = "u"
            v = ScalarField(grid, v_data)
            v.label = "v"

            return FieldCollection([u, v])

        # For gaussian-blobs, create localized seeds
        if ic_type == "gaussian-blobs":
            return self._blob_gray_scott_init(grid, ic_params)

        # Fallback to parent implementation
        return super().create_initial_state(grid, ic_type, ic_params)

    def _default_gray_scott_init(
        self,
        grid: CartesianGrid,
        params: dict[str, Any],
    ) -> FieldCollection:
        """Create default Gray-Scott initialization with center perturbation."""
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

        # Perturbation size
        r = params.get("perturbation_radius", 0.1) * min(Lx, Ly)

        # Create mask for perturbation region
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = dist < r

        # Initialize u = 0, v = 1 everywhere (reference default)
        u_data = np.zeros(grid.shape)
        v_data = np.ones(grid.shape)

        # Add perturbation in center - seed with some u
        u_data[mask] = 0.5

        # Add small noise
        noise = params.get("noise", 0.01)
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

    def _blob_gray_scott_init(
        self,
        grid: CartesianGrid,
        params: dict[str, Any],
    ) -> FieldCollection:
        """Create Gray-Scott init with Gaussian blob seeds."""
        num_blobs = params.get("num_blobs", 5)
        width = params.get("width", 0.05)

        # Get domain info
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[0]

        # Create coordinates
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Initialize u = 0, v = 1
        u_data = np.zeros(grid.shape)
        v_data = np.ones(grid.shape)

        # Add Gaussian blobs as seeds
        sigma = width * min(Lx, Ly)
        seed = params.get("seed")
        if seed is not None:
            np.random.seed(seed)
        for _ in range(num_blobs):
            cx = np.random.uniform(x_bounds[0] + sigma, x_bounds[1] - sigma)
            cy = np.random.uniform(y_bounds[0] + sigma, y_bounds[1] - sigma)

            blob = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))
            u_data += 0.5 * blob

        # Add small noise
        noise = params.get("noise", 0.01)
        u_data += noise * np.random.randn(*grid.shape)
        v_data += noise * np.random.randn(*grid.shape)

        # Clip
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
        b = parameters.get("b", 0.06)
        D = parameters.get("D", 2.0)

        return {
            "u": f"laplace(u) + u**2 * v - ({a} + {b}) * u",
            "v": f"{D} * laplace(v) - u**2 * v + {a} * (1 - v)",
        }

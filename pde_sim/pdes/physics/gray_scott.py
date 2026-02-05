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
                PDEParameter("a", "Feed rate"),
                PDEParameter("b", "Kill/removal rate"),
                PDEParameter("D", "Diffusion ratio (v diffuses D times faster than u)"),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Pearson (1993) Complex patterns in a simple system",
            supported_dimensions=[1, 2, 3],
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
        **kwargs,
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

        # For gaussian-blob, create localized seeds
        if ic_type == "gaussian-blob":
            return self._blob_gray_scott_init(grid, ic_params)

        # Fallback to parent implementation
        return super().create_initial_state(grid, ic_type, ic_params)

    def _default_gray_scott_init(
        self,
        grid: CartesianGrid,
        params: dict[str, Any],
    ) -> FieldCollection:
        """Create default Gray-Scott initialization with center perturbation."""
        seed = params.get("seed")
        rng = np.random.default_rng(seed)

        # Get domain info
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[0]

        # Create coordinates
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Perturbation center (randomize if not specified)
        cx = params.get("cx")
        cy = params.get("cy")
        if cx is None or cx == "random" or cy is None or cy == "random":
            raise ValueError("gray-scott default requires cx and cy (or random)")
        cx = x_bounds[0] + cx * Lx
        cy = y_bounds[0] + cy * Ly

        # Perturbation size
        r = params.get("perturbation_radius", 0.1) * min(Lx, Ly)

        # Create mask for perturbation region
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = dist < r

        # Initialize u = 0, v = 1 everywhere (reference default)
        u_data = np.zeros(grid.shape)
        v_data = np.ones(grid.shape)

        # Add perturbation - seed with some u
        u_data[mask] = 0.5

        # Add small noise
        noise = params.get("noise", 0.01)
        u_data += noise * rng.standard_normal(grid.shape)
        v_data += noise * rng.standard_normal(grid.shape)

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
        seed = params.get("seed")
        rng = np.random.default_rng(seed)

        if "positions" not in params:
            raise ValueError("gray-scott gaussian-blob requires positions or positions: random")
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

        positions = params["positions"]
        if len(positions) != num_blobs:
            raise ValueError(
                f"num_blobs={num_blobs} does not match positions length {len(positions)}"
            )

        # Add Gaussian blobs as seeds
        sigma = width * min(Lx, Ly)
        for cx_norm, cy_norm in positions:
            cx = x_bounds[0] + cx_norm * Lx
            cy = y_bounds[0] + cy_norm * Ly

            blob = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))
            u_data += 0.5 * blob

        # Add small noise
        noise = params.get("noise", 0.01)
        u_data += noise * rng.standard_normal(grid.shape)
        v_data += noise * rng.standard_normal(grid.shape)

        # Clip
        u_data = np.clip(u_data, 0, 1)
        v_data = np.clip(v_data, 0, 1)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

    def resolve_ic_params(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> dict[str, Any]:
        if ic_type in ("gray-scott-default", "default"):
            resolved = ic_params.copy()
            if "cx" not in resolved or "cy" not in resolved:
                raise ValueError("gray-scott default requires cx and cy (or random)")
            if resolved["cx"] == "random" or resolved["cy"] == "random":
                rng = np.random.default_rng(resolved.get("seed"))
                if resolved["cx"] == "random":
                    resolved["cx"] = rng.uniform(0.2, 0.8)
                if resolved["cy"] == "random":
                    resolved["cy"] = rng.uniform(0.2, 0.8)
            if resolved["cx"] is None or resolved["cy"] is None:
                raise ValueError("gray-scott default requires cx and cy (or random)")
            return resolved
        return super().resolve_ic_params(grid, ic_type, ic_params)

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

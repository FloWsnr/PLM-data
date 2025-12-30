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

    The Gray-Scott model describes pattern formation in chemical systems
    with autocatalytic reactions:

        du/dt = Du * laplace(u) - u*v^2 + F*(1-u)
        dv/dt = Dv * laplace(v) + u*v^2 - (F+k)*v

    where:
        - u is the concentration of the "feed" chemical
        - v is the concentration of the "catalyst" chemical
        - Du, Dv are diffusion coefficients
        - F is the feed rate
        - k is the kill rate

    The system exhibits a rich variety of patterns depending on F and k:
        - Spots, stripes, spirals, chaos, etc.

    Reference: Pearson, J. E. (1993). Complex patterns in a simple system.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="gray-scott",
            category="physics",
            description="Gray-Scott reaction-diffusion pattern formation",
            equations={
                "u": "Du * laplace(u) - u * v**2 + F * (1 - u)",
                "v": "Dv * laplace(v) + u * v**2 - (F + k) * v",
            },
            parameters=[
                PDEParameter(
                    name="F",
                    default=0.04,
                    description="Feed rate",
                    min_value=0.0,
                    max_value=0.1,
                ),
                PDEParameter(
                    name="k",
                    default=0.06,
                    description="Kill rate",
                    min_value=0.0,
                    max_value=0.1,
                ),
                PDEParameter(
                    name="Du",
                    default=0.16,
                    description="Diffusion coefficient for u",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="Dv",
                    default=0.08,
                    description="Diffusion coefficient for v",
                    min_value=0.01,
                    max_value=1.0,
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
            parameters: Dictionary with F, k, Du, Dv.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        F = parameters.get("F", 0.04)
        k = parameters.get("k", 0.06)
        Du = parameters.get("Du", 0.16)
        Dv = parameters.get("Dv", 0.08)

        return PDE(
            rhs={
                "u": f"{Du} * laplace(u) - u * v**2 + {F} * (1 - u)",
                "v": f"{Dv} * laplace(v) + u * v**2 - ({F} + {k}) * v",
            },
            bc=self._convert_bc(bc),
        )

    def _convert_bc(self, bc: dict[str, str]) -> str | dict:
        """Convert boundary condition config to py-pde format."""
        x_bc = bc.get("x", "periodic")
        y_bc = bc.get("y", "periodic")

        if x_bc == "periodic" and y_bc == "periodic":
            return "periodic"

        bc_map = {
            "periodic": "periodic",
            "neumann": "no-flux",
            "no-flux": "no-flux",
            "dirichlet": {"value": 0},
        }

        return bc_map.get(x_bc, "periodic")

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state for Gray-Scott.

        The default initialization for Gray-Scott is:
        - u = 1 everywhere with a perturbation region where u is lower
        - v = 0 everywhere with a perturbation region where v is higher

        This creates the "seed" for pattern formation.
        """
        # Check for per-field specifications
        if "u" in ic_params and isinstance(ic_params["u"], dict):
            # Use specified per-field ICs
            return super().create_initial_state(grid, ic_type, ic_params)

        # Default Gray-Scott initialization
        if ic_type == "gray-scott-default":
            return self._default_gray_scott_init(grid, ic_params)

        # For other IC types, create uniform backgrounds and add perturbation
        if ic_type in ("random-uniform", "random-gaussian"):
            # Create random perturbation for v
            v_field = create_initial_condition(grid, ic_type, ic_params)
            # Scale v to be small perturbations
            v_data = v_field.data * 0.1

            # u starts at 1.0 and is reduced where v is present
            u_data = np.ones(grid.shape) - v_data * 2

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

        # Initialize u = 1, v = 0 everywhere
        u_data = np.ones(grid.shape)
        v_data = np.zeros(grid.shape)

        # Add perturbation in center
        u_data[mask] = 0.5
        v_data[mask] = 0.25

        # Add small noise
        noise = params.get("noise", 0.01)
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

        # Initialize u = 1, v = 0
        u_data = np.ones(grid.shape)
        v_data = np.zeros(grid.shape)

        # Add Gaussian blobs as seeds
        sigma = width * min(Lx, Ly)
        for _ in range(num_blobs):
            cx = np.random.uniform(x_bounds[0] + sigma, x_bounds[1] - sigma)
            cy = np.random.uniform(y_bounds[0] + sigma, y_bounds[1] - sigma)

            blob = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))
            u_data -= 0.5 * blob
            v_data += 0.25 * blob

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
        F = parameters.get("F", 0.04)
        k = parameters.get("k", 0.06)
        Du = parameters.get("Du", 0.16)
        Dv = parameters.get("Dv", 0.08)

        return {
            "u": f"{Du} * laplace(u) - u * v**2 + {F} * (1 - u)",
            "v": f"{Dv} * laplace(v) + u * v**2 - ({F} + {k}) * v",
        }

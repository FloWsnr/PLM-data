"""Heat/diffusion equation and inhomogeneous variants."""

import math
from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from ..base import PDEMetadata, PDEParameter, ScalarPDEPreset
from .. import register_pde


@register_pde("heat")
class HeatPDE(ScalarPDEPreset):
    """Heat/diffusion equation.

    The heat equation is a fundamental parabolic PDE that describes
    how heat diffuses through a material over time:

        dT/dt = D_T * laplace(T)

    where T is the temperature and D_T is the thermal diffusivity.

    Based on visualpde.com heat equation formulation.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="heat",
            category="basic",
            description="Heat diffusion equation",
            equations={"T": "D_T * laplace(T)"},
            parameters=[
                PDEParameter(
                    name="D_T",
                    default=1.0,
                    description="Thermal diffusivity (diffusion coefficient)",
                    min_value=0.01,
                    max_value=10.0,
                ),
            ],
            num_fields=1,
            field_names=["T"],
            reference="https://visualpde.com/basic-pdes/heat-equation",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the heat equation PDE.

        Args:
            parameters: Dictionary containing 'D_T' (thermal diffusivity).
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        D_T = parameters.get("D_T", 1.0)

        # Convert BC to py-pde format
        bc_spec = self._convert_bc(bc)

        return PDE(
            rhs={"T": f"{D_T} * laplace(T)"},
            bc=bc_spec,
        )

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        D_T = parameters.get("D_T", 1.0)
        return {"T": f"{D_T} * laplace(T)"}


@register_pde("inhomogeneous-heat")
class InhomogeneousHeatPDE(ScalarPDEPreset):
    """Inhomogeneous heat equation with spatially varying source term.

    Based on the visualpde.com inhomogeneous heat equation:

        dT/dt = D * laplace(T) + f(x,y)

    where the source term f(x,y) is:

        f(x,y) = D * pi^2 * (n^2/L_x^2 + m^2/L_y^2) * cos(n*pi*x/L_x) * cos(m*pi*y/L_y)

    This specific form ensures the steady state solution is:
        T(x,y) = -cos(n*pi*x/L_x) * cos(m*pi*y/L_y)

    The integers n and m control the spatial frequency of the source pattern.
    For bounded solutions with Neumann BCs, the source must integrate to zero.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="inhomogeneous-heat",
            category="basic",
            description="Heat equation with spatially varying source term",
            equations={"T": "D * laplace(T) + f(x,y)"},
            parameters=[
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Diffusion coefficient",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="n",
                    default=4,
                    description="Spatial mode number in x direction",
                    min_value=1,
                    max_value=10,
                ),
                PDEParameter(
                    name="m",
                    default=4,
                    description="Spatial mode number in y direction",
                    min_value=1,
                    max_value=10,
                ),
            ],
            num_fields=1,
            field_names=["T"],
            reference="https://visualpde.com/basic-pdes/inhomogeneous-heat-equation",
            supported_dimensions=[2],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        n = int(parameters.get("n", 4))
        m = int(parameters.get("m", 4))

        # Get domain size from grid
        L_x = grid.axes_bounds[0][1] - grid.axes_bounds[0][0]
        L_y = grid.axes_bounds[1][1] - grid.axes_bounds[1][0]

        # Compute the source term coefficient
        # f(x,y) = D * pi^2 * (n^2/L_x^2 + m^2/L_y^2) * cos(n*pi*x/L_x) * cos(m*pi*y/L_y)
        coeff = D * math.pi**2 * (n**2 / L_x**2 + m**2 / L_y**2)

        # Create spatial source field
        x_coords = grid.cell_coords[..., 0]
        y_coords = grid.cell_coords[..., 1]
        source_data = coeff * np.cos(n * math.pi * x_coords / L_x) * np.cos(
            m * math.pi * y_coords / L_y
        )
        source_field = ScalarField(grid, source_data)

        return PDE(
            rhs={"T": f"{D} * laplace(T) + source"},
            bc=self._convert_bc(bc),
            consts={"source": source_field},
        )

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        D = parameters.get("D", 1.0)
        n = int(parameters.get("n", 4))
        m = int(parameters.get("m", 4))
        return {
            "T": f"{D} * laplace(T) + f(x,y)",
            "f(x,y)": f"D*pi^2*(n^2/L_x^2 + m^2/L_y^2)*cos(n*pi*x/L_x)*cos(m*pi*y/L_y) with n={n}, m={m}",
        }


@register_pde("inhomogeneous-diffusion-heat")
class InhomogeneousDiffusionHeatPDE(ScalarPDEPreset):
    """Heat equation with spatially-varying diffusion coefficient.

    Based on the visualpde.com inhomogeneous diffusion heat equation:

        dT/dt = div(g(x,y) * grad(T))

    where the diffusion coefficient g(x,y) has radial variation:

        g(x,y) = D * (1 + E * cos(n*pi*r/sqrt(L_x*L_y)))
        r = sqrt((x - L_x/2)^2 + (y - L_y/2)^2)

    The divergence expands to:
        dT/dt = g * laplace(T) + dg_dx * d_dx(T) + dg_dy * d_dy(T)

    This creates radial bands of alternating high/low thermal conductivity,
    causing heat to partition into regions bounded by conductivity maxima.

    Physical applications:
        - Heat flow through composite/layered materials
        - Geothermal modeling in heterogeneous rock
        - Thermal management in non-uniform materials
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="inhomogeneous-diffusion-heat",
            category="basic",
            description="Heat equation with spatially-varying diffusion coefficient",
            equations={"T": "div(g(x,y) * grad(T))"},
            parameters=[
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Base diffusion coefficient",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="E",
                    default=0.99,
                    description="Modulation amplitude (0 to 1)",
                    min_value=0.0,
                    max_value=0.999,
                ),
                PDEParameter(
                    name="n",
                    default=40,
                    description="Number of radial oscillations",
                    min_value=1,
                    max_value=50,
                ),
            ],
            num_fields=1,
            field_names=["T"],
            reference="https://visualpde.com/basic-pdes/inhomogeneous-diffusion",
            supported_dimensions=[2],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the inhomogeneous diffusion heat equation PDE.

        Args:
            parameters: Dictionary with D, E, n coefficients.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        D = parameters.get("D", 1.0)
        E = parameters.get("E", 0.99)
        n = int(parameters.get("n", 40))

        # Get domain size from grid
        L_x = grid.axes_bounds[0][1] - grid.axes_bounds[0][0]
        L_y = grid.axes_bounds[1][1] - grid.axes_bounds[1][0]
        L = math.sqrt(L_x * L_y)  # Characteristic length

        # Get spatial coordinates
        x_coords = grid.cell_coords[..., 0]
        y_coords = grid.cell_coords[..., 1]

        # Center of domain
        cx = (grid.axes_bounds[0][0] + grid.axes_bounds[0][1]) / 2
        cy = (grid.axes_bounds[1][0] + grid.axes_bounds[1][1]) / 2

        # Compute r = sqrt((x - cx)^2 + (y - cy)^2)
        dx = x_coords - cx
        dy = y_coords - cy
        r = np.sqrt(dx**2 + dy**2)

        # Compute g(x,y) = D * (1 + E * cos(n*pi*r/L))
        arg = n * math.pi * r / L
        cos_term = np.cos(arg)
        g_data = D * (1 + E * cos_term)
        g_field = ScalarField(grid, g_data)

        # Compute dg/dr = -D * E * (n*pi/L) * sin(n*pi*r/L)
        sin_term = np.sin(arg)
        dg_dr = -D * E * (n * math.pi / L) * sin_term

        # Compute dg/dx = dg/dr * dr/dx = dg/dr * (x - cx) / r
        # Compute dg/dy = dg/dr * dr/dy = dg/dr * (y - cy) / r
        # Handle r=0 case to avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            dg_dx_data = np.where(r > 1e-10, dg_dr * dx / r, 0.0)
            dg_dy_data = np.where(r > 1e-10, dg_dr * dy / r, 0.0)

        dg_dx_field = ScalarField(grid, dg_dx_data)
        dg_dy_field = ScalarField(grid, dg_dy_data)

        # div(g * grad(T)) = g * laplace(T) + dg_dx * d_dx(T) + dg_dy * d_dy(T)
        bc_spec = self._convert_bc(bc)

        return PDE(
            rhs={"T": "g * laplace(T) + dg_dx * d_dx(T) + dg_dy * d_dy(T)"},
            bc=bc_spec,
            consts={
                "g": g_field,
                "dg_dx": dg_dx_field,
                "dg_dy": dg_dy_field,
            },
        )

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        D = parameters.get("D", 1.0)
        E = parameters.get("E", 0.99)
        n = int(parameters.get("n", 40))
        return {
            "T": "div(g(x,y) * grad(T))",
            "g(x,y)": f"D*(1+E*cos(n*pi*r/L)) with D={D}, E={E}, n={n}",
        }


@register_pde("blob-diffusion-heat")
class BlobDiffusionHeatPDE(ScalarPDEPreset):
    """Heat equation with blob-based spatially-varying diffusion coefficient.

    Similar to inhomogeneous-diffusion-heat but uses gaussian blobs
    instead of radial cosine pattern for the diffusion coefficient:

        dT/dt = div(g(x,y) * grad(T))

    where the diffusion coefficient is a sum of gaussian blobs:

        g(x,y) = D_min + sum_i (D_max - D_min) * exp(-|r - r_i|^2 / (2*sigma^2))

    This creates randomly distributed regions of high thermal conductivity
    (at blob centers) surrounded by low conductivity regions.

    Physical applications:
        - Heat flow through materials with inclusions
        - Porous media with varying permeability
        - Biological tissue with heterogeneous thermal properties
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="blob-diffusion-heat",
            category="basic",
            description="Heat equation with blob-based varying diffusion",
            equations={"T": "div(g(x,y) * grad(T))"},
            parameters=[
                PDEParameter(
                    name="D_min",
                    default=0.1,
                    description="Minimum diffusion coefficient (background)",
                    min_value=0.01,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="D_max",
                    default=2.0,
                    description="Maximum diffusion coefficient (at blob centers)",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="n_blobs",
                    default=5,
                    description="Number of diffusion blobs",
                    min_value=1,
                    max_value=20,
                ),
                PDEParameter(
                    name="sigma",
                    default=0.1,
                    description="Blob size (fraction of domain)",
                    min_value=0.02,
                    max_value=0.3,
                ),
                PDEParameter(
                    name="seed",
                    default=42,
                    description="Random seed for blob placement",
                    min_value=0,
                    max_value=99999,
                ),
            ],
            num_fields=1,
            field_names=["T"],
            reference="https://visualpde.com/basic-pdes/inhomogeneous-diffusion",
            supported_dimensions=[2],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the blob diffusion heat equation PDE."""
        D_min = parameters.get("D_min", 0.1)
        D_max = parameters.get("D_max", 2.0)
        n_blobs = int(parameters.get("n_blobs", 5))
        sigma = parameters.get("sigma", 0.1)
        seed = int(parameters.get("seed", 42))

        # Get domain size from grid
        L_x = grid.axes_bounds[0][1] - grid.axes_bounds[0][0]
        L_y = grid.axes_bounds[1][1] - grid.axes_bounds[1][0]
        x_min, x_max = grid.axes_bounds[0]
        y_min, y_max = grid.axes_bounds[1]

        # Get spatial coordinates
        x_coords = grid.cell_coords[..., 0]
        y_coords = grid.cell_coords[..., 1]

        # Generate random blob centers
        rng = np.random.default_rng(seed)
        blob_x = rng.uniform(x_min + 0.1 * L_x, x_max - 0.1 * L_x, n_blobs)
        blob_y = rng.uniform(y_min + 0.1 * L_y, y_max - 0.1 * L_y, n_blobs)

        # Compute sigma in physical units
        sigma_phys = sigma * math.sqrt(L_x * L_y)

        # Build diffusion coefficient field as sum of gaussians
        g_data = np.full(grid.shape, D_min, dtype=float)
        dg_dx_data = np.zeros(grid.shape, dtype=float)
        dg_dy_data = np.zeros(grid.shape, dtype=float)

        for i in range(n_blobs):
            dx = x_coords - blob_x[i]
            dy = y_coords - blob_y[i]
            r2 = dx**2 + dy**2
            gaussian = np.exp(-r2 / (2 * sigma_phys**2))

            # Add to g
            g_data += (D_max - D_min) * gaussian

            # Compute gradient contribution
            # d/dx[exp(-r^2/(2*s^2))] = -x/s^2 * exp(...)
            dg_dx_data += (D_max - D_min) * gaussian * (-dx / sigma_phys**2)
            dg_dy_data += (D_max - D_min) * gaussian * (-dy / sigma_phys**2)

        g_field = ScalarField(grid, g_data)
        dg_dx_field = ScalarField(grid, dg_dx_data)
        dg_dy_field = ScalarField(grid, dg_dy_data)

        # div(g * grad(T)) = g * laplace(T) + dg_dx * d_dx(T) + dg_dy * d_dy(T)
        bc_spec = self._convert_bc(bc)

        return PDE(
            rhs={"T": "g * laplace(T) + dg_dx * d_dx(T) + dg_dy * d_dy(T)"},
            bc=bc_spec,
            consts={
                "g": g_field,
                "dg_dx": dg_dx_field,
                "dg_dy": dg_dy_field,
            },
        )

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        D_min = parameters.get("D_min", 0.1)
        D_max = parameters.get("D_max", 2.0)
        n_blobs = int(parameters.get("n_blobs", 5))
        sigma = parameters.get("sigma", 0.1)
        return {
            "T": "div(g(x,y) * grad(T))",
            "g(x,y)": f"D_min + sum of {n_blobs} gaussian blobs, D_min={D_min}, D_max={D_max}, sigma={sigma}",
        }

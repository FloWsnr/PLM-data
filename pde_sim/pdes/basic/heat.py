"""Heat/diffusion equation."""

import math
from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from ..base import PDEMetadata, PDEParameter, ScalarPDEPreset
from .. import register_pde


@register_pde("heat")
class HeatPDE(ScalarPDEPreset):
    """Heat/diffusion equation.

    The heat equation describes the diffusion of heat (or any quantity
    that diffuses) over time:

        dT/dt = D * laplace(T)

    where T is the temperature (or concentration) and D is the diffusion
    coefficient.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="heat",
            category="basic",
            description="Heat diffusion equation",
            equations={"T": "D * laplace(T)"},
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.1,
                    description="Diffusion coefficient",
                    min_value=0.01,
                    max_value=0.5,
                ),
            ],
            num_fields=1,
            field_names=["T"],
            reference="Basic parabolic PDE",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the heat equation PDE.

        Args:
            parameters: Dictionary containing 'D' (diffusion coefficient).
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        D = parameters.get("D", 1.0)

        # Convert BC to py-pde format
        bc_spec = self._convert_bc(bc)

        return PDE(
            rhs={"T": f"{D} * laplace(T)"},
            bc=bc_spec,
        )

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        D = parameters.get("D", 1.0)
        return {"T": f"{D} * laplace(T)"}


@register_pde("inhomogeneous-heat")
class InhomogeneousHeatPDE(ScalarPDEPreset):
    """Inhomogeneous heat equation with spatially varying source term.

    Based on the visualpde.com inhomogeneous heat equation:

        dT/dt = D * laplace(T) + f(x,y)

    where the source term f(x,y) is:

        f(x,y) = D * pi^2 * (n^2 + m^2) / L^2 * cos(n*pi*x/L) * cos(m*pi*y/L)

    This creates oscillating heat sources and sinks that average to zero
    over the domain, ensuring bounded solutions with Neumann BCs.

    The integers n and m control the spatial frequency of the source pattern.
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
                    default=0.1,
                    description="Diffusion coefficient",
                    min_value=0.01,
                    max_value=0.5,
                ),
                PDEParameter(
                    name="n",
                    default=2,
                    description="Spatial mode number in x direction",
                    min_value=1,
                    max_value=5,
                ),
                PDEParameter(
                    name="m",
                    default=2,
                    description="Spatial mode number in y direction",
                    min_value=1,
                    max_value=5,
                ),
            ],
            num_fields=1,
            field_names=["T"],
            reference="https://visualpde.com/basic-pdes/inhomogeneous-heat-equation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 0.1)
        n = int(parameters.get("n", 2))
        m = int(parameters.get("m", 2))

        # Get domain size from grid (assuming square domain)
        L = grid.axes_bounds[0][1] - grid.axes_bounds[0][0]

        # Compute the source term coefficient
        # f(x,y) = D * pi^2 * (n^2 + m^2) / L^2 * cos(n*pi*x/L) * cos(m*pi*y/L)
        coeff = D * math.pi**2 * (n**2 + m**2) / L**2

        # Create spatial source field
        # For py-pde, create the source as a constant field
        x_coords = grid.cell_coords[..., 0]
        y_coords = grid.cell_coords[..., 1]
        source_data = coeff * np.cos(n * math.pi * x_coords / L) * np.cos(
            m * math.pi * y_coords / L
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
        D = parameters.get("D", 0.1)
        n = int(parameters.get("n", 2))
        m = int(parameters.get("m", 2))
        return {
            "T": f"{D} * laplace(T) + f(x,y)",
            "f(x,y)": f"D*π²*(n²+m²)/L² * cos(nπx/L)*cos(mπy/L) with n={n}, m={m}",
        }

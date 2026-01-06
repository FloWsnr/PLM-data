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

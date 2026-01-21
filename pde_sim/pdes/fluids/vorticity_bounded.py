"""Vorticity equation for bounded domain with oscillatory initial conditions."""

import math
from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.core.config import BoundaryConfig

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("vorticity-bounded")
class VorticityBoundedPDE(MultiFieldPDEPreset):
    """Vorticity in bounded domain with oscillatory initial conditions.

    Demonstrates decay of oscillatory vorticity in a bounded domain.
    Uses different boundary conditions for different fields.

    Vorticity transport:
        d(omega)/dt = nu * laplace(omega) - d_dy(psi) * d_dx(omega) + d_dx(psi) * d_dy(omega)

    Stream function (parabolic relaxation of Poisson equation):
        epsilon * d(psi)/dt = laplace(psi) + omega

    Passive scalar transport:
        dS/dt = D * laplace(S) - d_dy(psi) * d_dx(S) + d_dx(psi) * d_dy(S)

    Boundary conditions:
        - omega: Dirichlet=0 on left/right, periodic top/bottom
        - S: Neumann=0 on left/right, periodic top/bottom
        - psi: periodic everywhere

    Initial condition:
        omega = A * cos(k*pi*x/L_x) * cos(k*pi*y/L_y)
        where A = 0.005 * k^1.5, k=51

    Reference:
        https://visualpde.com/fluids/vorticity-bounded
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="vorticity-bounded",
            category="fluids",
            description="Vorticity in bounded domain (oscillatory decay)",
            equations={
                "omega": "nu * laplace(omega) - d_dy(psi) * d_dx(omega) + d_dx(psi) * d_dy(omega)",
                "psi": "(laplace(psi) + omega) / epsilon",
                "S": "D * laplace(S) - d_dy(psi) * d_dx(S) + d_dx(psi) * d_dy(S)",
            },
            parameters=[
                PDEParameter(
                    name="nu",
                    default=0.1,
                    description="Kinematic viscosity",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="epsilon",
                    default=0.05,
                    description="Relaxation parameter for Poisson solver",
                    min_value=0.01,
                    max_value=0.5,
                ),
                PDEParameter(
                    name="D",
                    default=0.05,
                    description="Passive scalar diffusion coefficient",
                    min_value=0.001,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="k",
                    default=51,
                    description="Wavenumber for oscillatory initial condition",
                    min_value=1,
                    max_value=100,
                ),
            ],
            num_fields=3,
            field_names=["omega", "psi", "S"],
            reference="VisualPDE NavierStokesVorticityBounded",
            supported_dimensions=[2],
        )

    def get_default_bc(self) -> BoundaryConfig:
        """Return default per-field boundary conditions."""
        return BoundaryConfig(
            x_minus="periodic",
            x_plus="periodic",
            y_minus="periodic",
            y_plus="periodic",
            fields={
                "omega": {
                    "x-": "dirichlet:0",
                    "x+": "dirichlet:0",
                    # y inherits from parent (periodic)
                },
                "S": {
                    "x-": "neumann:0",
                    "x+": "neumann:0",
                    # y inherits from parent (periodic)
                },
                # psi inherits default periodic for all boundaries
            },
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: Any,
        grid: CartesianGrid,
    ) -> PDE:
        nu = parameters.get("nu", 0.1)
        epsilon = parameters.get("epsilon", 0.05)
        D = parameters.get("D", 0.05)

        inv_eps = 1.0 / epsilon

        # If no boundary conditions specified, use defaults for bounded domain
        if bc is None or (isinstance(bc, dict) and not bc):
            bc = self.get_default_bc()

        return PDE(
            rhs={
                "omega": f"{nu} * laplace(omega) - d_dy(psi) * d_dx(omega) + d_dx(psi) * d_dy(omega)",
                "psi": f"{inv_eps} * (laplace(psi) + omega)",
                "S": f"{D} * laplace(S) - d_dy(psi) * d_dx(S) + d_dx(psi) * d_dy(S)",
            },
            **self._get_pde_bc_kwargs(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create oscillatory initial conditions.

        Default: omega = A * cos(k*pi*x/L_x) * cos(k*pi*y/L_y)
        where A = 0.005 * k^1.5
        """
        # Get domain bounds from grid
        x_min, x_max = grid.axes_bounds[0]
        y_min, y_max = grid.axes_bounds[1]
        L_x = x_max - x_min
        L_y = y_max - y_min

        # Get parameters
        k = int(ic_params.get("k", 51))
        A = 0.005 * (k ** 1.5)

        # Create coordinate arrays
        x = np.linspace(x_min, x_max, grid.shape[0])
        y = np.linspace(y_min, y_max, grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Oscillatory vorticity: A * cos(k*pi*x/L_x) * cos(k*pi*y/L_y)
        omega_data = A * np.cos(k * math.pi * X / L_x) * np.cos(k * math.pi * Y / L_y)

        # Stream function starts at zero (Poisson solver will equilibrate)
        psi_data = np.zeros_like(omega_data)

        # Passive scalar: gradient from 1 (left) to 0 (right)
        x_norm = (X - x_min) / L_x
        S_data = 1.0 - x_norm

        omega = ScalarField(grid, omega_data)
        omega.label = "omega"
        psi = ScalarField(grid, psi_data)
        psi.label = "psi"
        S = ScalarField(grid, S_data)
        S.label = "S"

        return FieldCollection([omega, psi, S])

"""Vorticity equation for 2D incompressible flow."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("vorticity")
class VorticityPDE(MultiFieldPDEPreset):
    """2D Vorticity-streamfunction formulation of incompressible flow.

    The vorticity equation reformulates the Navier-Stokes equations by
    focusing on vorticity (curl of velocity) rather than velocity itself.

    Vorticity transport:
        d(omega)/dt = nu * laplace(omega) - d_dy(psi) * d_dx(omega) + d_dx(psi) * d_dy(omega)

    Stream function (parabolic relaxation of Poisson equation):
        epsilon * d(psi)/dt = laplace(psi) + omega

    Passive scalar transport:
        dS/dt = D * laplace(S) - (d_dy(psi) * d_dx(S) - d_dx(psi) * d_dy(S))

    Velocity from stream function:
        u = d_dy(psi), v = -d_dx(psi)

    where:
        - omega is vorticity (curl of velocity)
        - psi is stream function
        - S is passive scalar for visualization
        - nu is kinematic viscosity
        - epsilon is relaxation parameter for Poisson solver
        - D is passive scalar diffusion coefficient
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="vorticity",
            category="fluids",
            description="2D vorticity-streamfunction formulation",
            equations={
                "omega": "nu * laplace(omega) - d_dy(psi) * d_dx(omega) + d_dx(psi) * d_dy(omega)",
                "psi": "(laplace(psi) + omega) / epsilon",
                "S": "D * laplace(S) - d_dy(psi) * d_dx(S) + d_dx(psi) * d_dy(S)",
            },
            parameters=[
                PDEParameter("nu", "Kinematic viscosity"),
                PDEParameter("epsilon", "Relaxation parameter for Poisson solver"),
                PDEParameter("D", "Passive scalar diffusion coefficient"),
            ],
            num_fields=3,
            field_names=["omega", "psi", "S"],
            reference="Vorticity-streamfunction formulation (Chemin)",
            supported_dimensions=[2],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
        init_params: dict[str, Any] | None = None,
    ) -> PDE:
        nu = parameters.get("nu", 0.05)
        epsilon = parameters.get("epsilon", 0.05)
        D = parameters.get("D", 0.05)

        inv_eps = 1.0 / epsilon

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
        """Create initial vorticity, stream function, and passive scalar."""
        # Get domain bounds from grid
        x_min, x_max = grid.axes_bounds[0]
        y_min, y_max = grid.axes_bounds[1]
        L_x = x_max - x_min
        L_y = y_max - y_min

        if ic_type in ("vorticity-default", "vortex-pair"):
            # Two counter-rotating vortices
            x0_1, y0_1 = ic_params.get("x1", 0.35), ic_params.get("y1", 0.5)
            x0_2, y0_2 = ic_params.get("x2", 0.65), ic_params.get("y2", 0.5)
            strength = ic_params.get("strength", 10.0)
            radius = ic_params.get("radius", 0.1)

            x, y = np.meshgrid(
                np.linspace(x_min, x_max, grid.shape[0]),
                np.linspace(y_min, y_max, grid.shape[1]),
                indexing="ij",
            )

            # Normalize coordinates
            x_norm = (x - x_min) / L_x
            y_norm = (y - y_min) / L_y

            # Gaussian vortices
            r1_sq = (x_norm - x0_1) ** 2 + (y_norm - y0_1) ** 2
            r2_sq = (x_norm - x0_2) ** 2 + (y_norm - y0_2) ** 2

            omega_data = strength * np.exp(-r1_sq / (2 * radius**2)) - strength * np.exp(
                -r2_sq / (2 * radius**2)
            )
            psi_data = np.zeros_like(omega_data)
            # Passive scalar: gradient from 1 (left) to 0 (right), matching Visual PDE
            S_data = 1.0 - x_norm

        elif ic_type == "single-vortex":
            x0 = ic_params.get("x0", 0.5)
            y0 = ic_params.get("y0", 0.5)
            strength = ic_params.get("strength", 10.0)
            radius = ic_params.get("radius", 0.15)

            x, y = np.meshgrid(
                np.linspace(x_min, x_max, grid.shape[0]),
                np.linspace(y_min, y_max, grid.shape[1]),
                indexing="ij",
            )

            # Normalize coordinates
            x_norm = (x - x_min) / L_x
            y_norm = (y - y_min) / L_y

            r_sq = (x_norm - x0) ** 2 + (y_norm - y0) ** 2
            omega_data = strength * np.exp(-r_sq / (2 * radius**2))
            psi_data = np.zeros_like(omega_data)
            # Passive scalar: gradient from 1 (left) to 0 (right), matching Visual PDE
            S_data = 1.0 - x_norm

        elif ic_type == "oscillatory":
            # Oscillatory vorticity field (NavierStokesVorticityBounded variant)
            # Initial condition: A*cos(k*pi*x/L_x)*cos(k*pi*y/L_y) where A = 0.005*k^1.5
            k = ic_params.get("k", 8)  # Wavenumber (default 8 for visible pattern)
            amplitude = ic_params.get("amplitude", 0.005 * k**1.5)

            x, y = np.meshgrid(
                np.linspace(x_min, x_max, grid.shape[0]),
                np.linspace(y_min, y_max, grid.shape[1]),
                indexing="ij",
            )

            # Normalize coordinates to [0, 1]
            x_norm = (x - x_min) / L_x
            y_norm = (y - y_min) / L_y

            # Oscillatory vorticity pattern
            omega_data = amplitude * np.cos(k * np.pi * x_norm) * np.cos(k * np.pi * y_norm)
            psi_data = np.zeros_like(omega_data)
            S_data = 1.0 - x_norm

        elif ic_type == "quad-vortex":
            # Four vortices in a square arrangement
            strength = ic_params.get("strength", 20.0)
            radius = ic_params.get("radius", 0.08)
            spacing = ic_params.get("spacing", 0.2)  # Distance from center

            x, y = np.meshgrid(
                np.linspace(x_min, x_max, grid.shape[0]),
                np.linspace(y_min, y_max, grid.shape[1]),
                indexing="ij",
            )

            x_norm = (x - x_min) / L_x
            y_norm = (y - y_min) / L_y

            # Four vortex positions around center with alternating signs
            centers = [
                (0.5 - spacing, 0.5 - spacing, 1),   # bottom-left, positive
                (0.5 + spacing, 0.5 - spacing, -1),  # bottom-right, negative
                (0.5 - spacing, 0.5 + spacing, -1),  # top-left, negative
                (0.5 + spacing, 0.5 + spacing, 1),   # top-right, positive
            ]

            omega_data = np.zeros_like(x_norm)
            for cx, cy, sign in centers:
                r_sq = (x_norm - cx) ** 2 + (y_norm - cy) ** 2
                omega_data += sign * strength * np.exp(-r_sq / (2 * radius**2))

            psi_data = np.zeros_like(omega_data)
            S_data = 1.0 - x_norm

        else:
            # Default: use standard IC generator for omega, zeros for psi
            omega_field = create_initial_condition(grid, ic_type, ic_params)
            omega_data = omega_field.data
            psi_data = np.zeros_like(omega_data)
            # Passive scalar: gradient from 1 (left) to 0 (right), matching Visual PDE
            x, y = np.meshgrid(
                np.linspace(x_min, x_max, grid.shape[0]),
                np.linspace(y_min, y_max, grid.shape[1]),
                indexing="ij",
            )
            x_norm = (x - x_min) / L_x
            S_data = 1.0 - x_norm

        omega = ScalarField(grid, omega_data)
        omega.label = "omega"
        psi = ScalarField(grid, psi_data)
        psi.label = "psi"
        S = ScalarField(grid, S_data)
        S.label = "S"

        return FieldCollection([omega, psi, S])

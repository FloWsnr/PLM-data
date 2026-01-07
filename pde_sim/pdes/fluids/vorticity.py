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
                PDEParameter(
                    name="nu",
                    default=0.05,
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
            ],
            num_fields=3,
            field_names=["omega", "psi", "S"],
            reference="Vorticity-streamfunction formulation (Chemin)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
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

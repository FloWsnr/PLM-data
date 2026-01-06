"""Rayleigh-Benard thermal convection."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("thermal-convection")
class ThermalConvectionPDE(MultiFieldPDEPreset):
    """Rayleigh-Benard thermal convection (Boussinesq model).

    This system models buoyancy-driven fluid motion from temperature gradients.
    When a fluid layer is heated from below, it becomes unstable and develops
    organized convection cells (Benard cells).

    Vorticity evolution with buoyancy forcing:
        d(omega)/dt = nu * laplace(omega) - d_dy(psi) * d_dx(omega) + d_dx(psi) * d_dy(omega) + d_dx(b)

    Stream function (relaxation form):
        epsilon * d(psi)/dt = laplace(psi) + omega

    Temperature perturbation:
        db/dt = kappa * laplace(b) - (d_dy(psi) * d_dx(b) - d_dx(psi) * d_dy(b))

    Velocity from stream function:
        u = d_dy(psi), v = -d_dx(psi)

    where:
        - omega is vorticity
        - psi is stream function
        - b is temperature perturbation from top boundary
        - nu is kinematic viscosity
        - epsilon is relaxation parameter
        - kappa is thermal diffusivity
        - T_b is bottom boundary temperature flux
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="thermal-convection",
            category="fluids",
            description="Rayleigh-Benard thermal convection",
            equations={
                "omega": "nu * laplace(omega) - d_dy(psi) * d_dx(omega) + d_dx(psi) * d_dy(omega) + d_dx(b)",
                "psi": "(laplace(psi) + omega) / epsilon",
                "b": "kappa * laplace(b) - d_dy(psi) * d_dx(b) + d_dx(psi) * d_dy(b)",
            },
            parameters=[
                PDEParameter(
                    name="nu",
                    default=0.2,
                    description="Kinematic viscosity",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="epsilon",
                    default=0.05,
                    description="Relaxation parameter for stream function",
                    min_value=0.01,
                    max_value=0.5,
                ),
                PDEParameter(
                    name="kappa",
                    default=0.5,
                    description="Thermal diffusivity",
                    min_value=0.01,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="T_b",
                    default=0.08,
                    description="Bottom boundary temperature flux",
                    min_value=0.01,
                    max_value=1.0,
                ),
            ],
            num_fields=3,
            field_names=["omega", "psi", "b"],
            reference="Rayleigh-Benard convection (Boussinesq, Oberbeck)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        nu = parameters.get("nu", 0.2)
        epsilon = parameters.get("epsilon", 0.05)
        kappa = parameters.get("kappa", 0.5)

        inv_eps = 1.0 / epsilon

        return PDE(
            rhs={
                "omega": f"{nu} * laplace(omega) - d_dy(psi) * d_dx(omega) + d_dx(psi) * d_dy(omega) + d_dx(b)",
                "psi": f"{inv_eps} * (laplace(psi) + omega)",
                "b": f"{kappa} * laplace(b) - d_dy(psi) * d_dx(b) + d_dx(psi) * d_dy(b)",
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial temperature perturbation with small noise."""
        # Get domain bounds from grid
        x_min, x_max = grid.axes_bounds[0]
        y_min, y_max = grid.axes_bounds[1]
        L_y = y_max - y_min

        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.01)

        x, y = np.meshgrid(
            np.linspace(x_min, x_max, grid.shape[0]),
            np.linspace(y_min, y_max, grid.shape[1]),
            indexing="ij",
        )

        # Normalize y coordinate
        y_norm = (y - y_min) / L_y

        if ic_type in ("thermal-convection-default", "default", "linear-gradient"):
            # Small random perturbations (instability seeds)
            omega_data = noise * np.random.randn(*grid.shape)
            psi_data = np.zeros_like(omega_data)
            # Small temperature perturbations near bottom (where heating occurs)
            b_data = noise * np.random.randn(*grid.shape) * (1.0 - y_norm)

        elif ic_type == "warm-blob":
            # Localized warm region
            x0 = ic_params.get("x0", 0.5)
            y0 = ic_params.get("y0", 0.2)
            amplitude = ic_params.get("amplitude", 0.5)
            radius = ic_params.get("radius", 0.1)

            # Normalize x coordinate
            x_norm = (x - x_min) / (x_max - x_min)

            r_sq = (x_norm - x0) ** 2 + (y_norm - y0) ** 2
            b_data = amplitude * np.exp(-r_sq / (2 * radius**2))
            omega_data = noise * np.random.randn(*grid.shape)
            psi_data = np.zeros_like(omega_data)

        else:
            # Default: use standard IC generator for b, small noise for omega
            b_field = create_initial_condition(grid, ic_type, ic_params)
            b_data = b_field.data
            omega_data = noise * np.random.randn(*grid.shape)
            psi_data = np.zeros_like(omega_data)

        omega = ScalarField(grid, omega_data)
        omega.label = "omega"
        psi = ScalarField(grid, psi_data)
        psi.label = "psi"
        b = ScalarField(grid, b_data)
        b.label = "b"

        return FieldCollection([omega, psi, b])

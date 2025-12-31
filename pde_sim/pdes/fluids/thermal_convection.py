"""Rayleigh-Bénard thermal convection."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("thermal-convection")
class ThermalConvectionPDE(MultiFieldPDEPreset):
    """Rayleigh-Bénard thermal convection.

    Coupled temperature and stream function equations:

        dT/dt = kappa * laplace(T) - u * d_dx(T) - v * d_dy(T)
        dw/dt = nu * laplace(w) + Ra * d_dx(T)

    where:
        - T is temperature
        - w is vorticity
        - Ra is Rayleigh number (buoyancy driving)
        - kappa is thermal diffusivity
        - nu is kinematic viscosity

    Simplified version with explicit coupling.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="thermal-convection",
            category="fluids",
            description="Rayleigh-Bénard convection",
            equations={
                "T": "kappa * laplace(T)",
                "w": "nu * laplace(w) + Ra * d_dx(T)",
            },
            parameters=[
                PDEParameter(
                    name="kappa",
                    default=0.01,
                    description="Thermal diffusivity",
                    min_value=0.001,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="nu",
                    default=0.01,
                    description="Kinematic viscosity",
                    min_value=0.001,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="Ra",
                    default=100.0,
                    description="Rayleigh number",
                    min_value=1.0,
                    max_value=10000.0,
                ),
            ],
            num_fields=2,
            field_names=["T", "w"],
            reference="Rayleigh-Bénard convection rolls",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        kappa = parameters.get("kappa", 0.01)
        nu = parameters.get("nu", 0.01)
        Ra = parameters.get("Ra", 100.0)

        return PDE(
            rhs={
                "T": f"{kappa} * laplace(T)",
                "w": f"{nu} * laplace(w) + {Ra} * d_dx(T)",
            },
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial temperature gradient with perturbation."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.1)

        x, y = np.meshgrid(
            np.linspace(0, 1, grid.shape[0]),
            np.linspace(0, 1, grid.shape[1]),
            indexing="ij",
        )

        # Linear temperature profile (hot bottom, cold top) with perturbation
        T_data = 1.0 - y + noise * np.random.randn(*grid.shape)

        # Small initial vorticity
        w_data = noise * np.random.randn(*grid.shape)

        T = ScalarField(grid, T_data)
        T.label = "T"
        w = ScalarField(grid, w_data)
        w.label = "w"

        return FieldCollection([T, w])

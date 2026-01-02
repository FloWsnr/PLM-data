"""Coupled nonlinear oscillators (Van der Pol type)."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("oscillators")
class OscillatorsPDE(MultiFieldPDEPreset):
    """Coupled nonlinear oscillators (Van der Pol type).

    Spatially extended Van der Pol oscillator:

        du/dt = Du * laplace(u) + v
        dv/dt = Dv * laplace(v) + mu * (1 - u^2) * v - omega^2 * u

    where:
        - u is the position variable (renamed from x to avoid grid coordinate collision)
        - v is the velocity variable (renamed from y to avoid grid coordinate collision)
        - mu controls nonlinearity strength
        - omega is the natural frequency
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="oscillators",
            category="physics",
            description="Coupled Van der Pol oscillators",
            equations={
                "u": "Du * laplace(u) + v",
                "v": "Dv * laplace(v) + mu * (1 - u^2) * v - omega^2 * u",
            },
            parameters=[
                PDEParameter(
                    name="mu",
                    default=1.0,
                    description="Nonlinearity strength",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="omega",
                    default=1.0,
                    description="Natural frequency",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Du",
                    default=0.1,
                    description="Diffusion of u",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dv",
                    default=0.1,
                    description="Diffusion of v",
                    min_value=0.0,
                    max_value=10.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Van der Pol oscillator field",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        mu = parameters.get("mu", 1.0)
        omega = parameters.get("omega", 1.0)
        Du = parameters.get("Du", 0.1)
        Dv = parameters.get("Dv", 0.1)
        omega_sq = omega**2

        return PDE(
            rhs={
                "u": f"{Du} * laplace(u) + v",
                "v": f"{Dv} * laplace(v) + {mu} * (1 - u**2) * v - {omega_sq} * u",
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial oscillator states."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.5)

        u_data = noise * np.random.randn(*grid.shape)
        v_data = noise * np.random.randn(*grid.shape)

        u_field = ScalarField(grid, u_data)
        u_field.label = "u"
        v_field = ScalarField(grid, v_data)
        v_field.label = "v"

        return FieldCollection([u_field, v_field])

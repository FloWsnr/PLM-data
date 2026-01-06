"""Coupled nonlinear oscillators (Van der Pol type)."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("oscillators")
class OscillatorsPDE(MultiFieldPDEPreset):
    """Diffusively coupled Van der Pol oscillators.

    Based on visualpde.com formulation:

        du/dt = v
        dv/dt = D * (laplace(u) + eps * laplace(v)) + mu * (1 - u²) * v - u

    where:
        - u is the position variable
        - v is the velocity variable
        - D is the diffusion coupling strength
        - eps is artificial diffusion for numerical stability
        - mu controls nonlinearity strength (damping)

    The diffusive coupling connects oscillators at each spatial point,
    leading to interesting collective dynamics like spiral waves.

    Reference: https://visualpde.com/nonlinear-physics/oscillators
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="oscillators",
            category="physics",
            description="Diffusively coupled Van der Pol oscillators",
            equations={
                "u": "v",
                "v": "D * (laplace(u) + eps * laplace(v)) + mu * (1 - u^2) * v - u",
            },
            parameters=[
                PDEParameter(
                    name="mu",
                    default=1.0,
                    description="Nonlinearity strength (damping)",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="D",
                    default=0.5,
                    description="Diffusion coupling strength",
                    min_value=0.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="eps",
                    default=0.1,
                    description="Artificial diffusion (numerical stability)",
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="https://visualpde.com/nonlinear-physics/oscillators",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        mu = parameters.get("mu", 1.0)
        D = parameters.get("D", 0.5)
        eps = parameters.get("eps", 0.1)

        return PDE(
            rhs={
                "u": "v",
                "v": f"{D} * (laplace(u) + {eps} * laplace(v)) + {mu} * (1 - u**2) * v - u",
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

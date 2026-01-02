"""Brusselator reaction-diffusion system."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("brusselator")
class BrusselatorPDE(MultiFieldPDEPreset):
    """Brusselator reaction-diffusion system.

    The Brusselator model for chemical oscillations and patterns:

        du/dt = Du * laplace(u) + a - (b+1)*u + u²v
        dv/dt = Dv * laplace(v) + b*u - u²v

    Exhibits both oscillations and Turing patterns.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="brusselator",
            category="biology",
            description="Brusselator oscillating reaction-diffusion",
            equations={
                "u": "Du * laplace(u) + a - (b + 1) * u + u**2 * v",
                "v": "Dv * laplace(v) + b * u - u**2 * v",
            },
            parameters=[
                PDEParameter(
                    name="a",
                    default=1.0,
                    description="Parameter a",
                    min_value=0.1,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="b",
                    default=3.0,
                    description="Parameter b",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Du",
                    default=1.0,
                    description="Diffusion of u",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dv",
                    default=8.0,
                    description="Diffusion of v",
                    min_value=0.1,
                    max_value=100.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Brusselator chemical oscillator",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        a = parameters.get("a", 1.0)
        b = parameters.get("b", 3.0)
        Du = parameters.get("Du", 1.0)
        Dv = parameters.get("Dv", 8.0)

        return PDE(
            rhs={
                "u": f"{Du} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
                "v": f"{Dv} * laplace(v) + {b} * u - u**2 * v",
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state near homogeneous steady state."""
        a = ic_params.get("a", 1.0)
        b = ic_params.get("b", 3.0)
        noise = ic_params.get("noise", 0.1)

        # Steady state: u* = a, v* = b/a
        u_ss = a
        v_ss = b / a

        np.random.seed(ic_params.get("seed"))
        u_data = u_ss + noise * np.random.randn(*grid.shape)
        v_data = v_ss + noise * np.random.randn(*grid.shape)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

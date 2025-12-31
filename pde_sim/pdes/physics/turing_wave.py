"""Turing patterns with wave instability."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("turing-wave")
class TuringWavePDE(MultiFieldPDEPreset):
    """Turing patterns with wave instability.

    Reaction-diffusion system exhibiting both Turing and wave instabilities:

        du/dt = Du * laplace(u) + f(u, v)
        dv/dt = Dv * laplace(v) + g(u, v)

    with kinetics chosen to produce oscillating patterns.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="turing-wave",
            category="physics",
            description="Turing-wave pattern interaction",
            equations={
                "u": "Du * laplace(u) + a - u + u^2 * v - gamma * u",
                "v": "Dv * laplace(v) + b - u^2 * v + gamma * u",
            },
            parameters=[
                PDEParameter(
                    name="Du",
                    default=0.1,
                    description="Activator diffusion",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="Dv",
                    default=5.0,
                    description="Inhibitor diffusion",
                    min_value=0.1,
                    max_value=50.0,
                ),
                PDEParameter(
                    name="a",
                    default=0.1,
                    description="Activator production",
                    min_value=0.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="b",
                    default=0.9,
                    description="Inhibitor production",
                    min_value=0.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="gamma",
                    default=0.1,
                    description="Cross-coupling",
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Turing-Hopf interaction patterns",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Du = parameters.get("Du", 0.1)
        Dv = parameters.get("Dv", 5.0)
        a = parameters.get("a", 0.1)
        b = parameters.get("b", 0.9)
        gamma = parameters.get("gamma", 0.1)

        return PDE(
            rhs={
                "u": f"{Du} * laplace(u) + {a} - u + u**2 * v - {gamma} * u",
                "v": f"{Dv} * laplace(v) + {b} - u**2 * v + {gamma} * u",
            },
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state near steady state."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.05)

        u_data = 0.5 + noise * np.random.randn(*grid.shape)
        v_data = 0.5 + noise * np.random.randn(*grid.shape)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

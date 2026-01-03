"""Turing patterns with advection."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("advecting-patterns")
class AdvectingPatternsPDE(MultiFieldPDEPreset):
    """Turing patterns with advection.

    Reaction-diffusion with flow:

        du/dt = Du * laplace(u) - v_flow * d_dx(u) + f(u,v)
        dv/dt = Dv * laplace(v) - v_flow * d_dx(v) + g(u,v)

    Shows how flow affects pattern formation.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="advecting-patterns",
            category="physics",
            description="Advected Turing patterns",
            equations={
                "u": "Du * laplace(u) - v_flow * d_dx(u) + a - u + u^2 * v",
                "v": "Dv * laplace(v) - v_flow * d_dx(v) + b - u^2 * v",
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
                    max_value=20.0,
                ),
                PDEParameter(
                    name="v_flow",
                    default=0.5,
                    description="Advection velocity",
                    min_value=0.0,
                    max_value=2.0,
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
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Pattern formation in flows",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Du = parameters.get("Du", 0.1)
        Dv = parameters.get("Dv", 5.0)
        v_flow = parameters.get("v_flow", 0.5)
        a = parameters.get("a", 0.1)
        b = parameters.get("b", 0.9)

        u_rhs = f"{Du} * laplace(u) + {a} - u + u**2 * v"
        v_rhs = f"{Dv} * laplace(v) + {b} - u**2 * v"

        if v_flow != 0:
            u_rhs = f"{Du} * laplace(u) - {v_flow} * d_dx(u) + {a} - u + u**2 * v"
            v_rhs = f"{Dv} * laplace(v) - {v_flow} * d_dx(v) + {b} - u**2 * v"

        return PDE(
            rhs={"u": u_rhs, "v": v_rhs},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.05)

        u_data = 0.5 + noise * np.random.randn(*grid.shape)
        v_data = 0.5 + noise * np.random.randn(*grid.shape)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

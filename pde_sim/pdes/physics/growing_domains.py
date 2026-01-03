"""Reaction-diffusion on effectively growing domain."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("growing-domains")
class GrowingDomainsPDE(MultiFieldPDEPreset):
    """Reaction-diffusion on effectively growing domain.

    Simulated domain growth via dilution terms:

        du/dt = Du * laplace(u) + f(u,v) - rho * u
        dv/dt = Dv * laplace(v) + g(u,v) - rho * v

    where rho represents the dilution due to domain growth.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="growing-domains",
            category="physics",
            description="Patterns on growing domains",
            equations={
                "u": "Du * laplace(u) + a - u + u^2 * v - rho * u",
                "v": "Dv * laplace(v) + b - u^2 * v - rho * v",
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
                    name="rho",
                    default=0.01,
                    description="Growth/dilution rate",
                    min_value=0.0,
                    max_value=0.5,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Pattern formation on growing domains",
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
        rho = parameters.get("rho", 0.01)

        u_rhs = f"{Du} * laplace(u) + {a} - u + u**2 * v - {rho} * u"
        v_rhs = f"{Dv} * laplace(v) + {b} - u**2 * v - {rho} * v"

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
        """Create initial state with seed pattern."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.05)

        u_data = 0.5 + noise * np.random.randn(*grid.shape)
        v_data = 0.5 + noise * np.random.randn(*grid.shape)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

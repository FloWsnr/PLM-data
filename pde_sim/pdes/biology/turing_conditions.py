"""Demonstration of Turing pattern conditions."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("turing-conditions")
class TuringConditionsPDE(MultiFieldPDEPreset):
    """Demonstration of Turing pattern conditions.

    Two-species activator-inhibitor system with parameters
    chosen to satisfy Turing instability conditions:

        du/dt = Du * laplace(u) + f(u,v)
        dv/dt = Dv * laplace(v) + g(u,v)

    where f = a * u - b * v and g = c * u - d * v (linearized)
    Full: f = u * (a - u) - u * v,  g = u * v - d * v

    Turing conditions require Dv >> Du and specific kinetics.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="turing-conditions",
            category="biology",
            description="Turing instability demonstration",
            equations={
                "u": "Du * laplace(u) + u * (a - u) - u * v",
                "v": "Dv * laplace(v) + u * v - d * v",
            },
            parameters=[
                PDEParameter(
                    name="Du",
                    default=0.1,
                    description="Activator diffusion (small)",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="Dv",
                    default=10.0,
                    description="Inhibitor diffusion (large)",
                    min_value=1.0,
                    max_value=100.0,
                ),
                PDEParameter(
                    name="a",
                    default=1.0,
                    description="Activator growth rate",
                    min_value=0.1,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="d",
                    default=0.5,
                    description="Inhibitor decay rate",
                    min_value=0.1,
                    max_value=5.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Turing (1952) pattern formation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Du = parameters.get("Du", 0.1)
        Dv = parameters.get("Dv", 10.0)
        a = parameters.get("a", 1.0)
        d = parameters.get("d", 0.5)

        return PDE(
            rhs={
                "u": f"{Du} * laplace(u) + u * ({a} - u) - u * v",
                "v": f"{Dv} * laplace(v) + u * v - {d} * v",
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create small perturbation around steady state."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.05)

        # Near uniform with small perturbations
        u_data = 0.5 + noise * np.random.randn(*grid.shape)
        v_data = 0.5 + noise * np.random.randn(*grid.shape)
        u_data = np.clip(u_data, 0.01, 2.0)
        v_data = np.clip(v_data, 0.01, 2.0)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

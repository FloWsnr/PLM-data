"""Klausmeier vegetation model for semi-arid ecosystems."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("vegetation")
class VegetationPDE(MultiFieldPDEPreset):
    """Klausmeier vegetation model for semi-arid ecosystems.

    Models water-vegetation dynamics on sloped terrain:

        dw/dt = a - w - w * n^2 + v * d_dx(w)
        dn/dt = Dn * laplace(n) + w * n^2 - m * n

    where:
        - w is water density
        - n is vegetation (plant) density
        - a is rainfall rate
        - v is water flow velocity (downhill)
        - m is plant mortality rate
        - Dn is plant dispersal

    Produces banded vegetation patterns on hillslopes.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="vegetation",
            category="biology",
            description="Klausmeier vegetation-water dynamics",
            equations={
                "w": "a - w - w * n**2 + v * d_dx(w)",
                "n": "Dn * laplace(n) + w * n**2 - m * n",
            },
            parameters=[
                PDEParameter(
                    name="a",
                    default=2.0,
                    description="Rainfall rate",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="v",
                    default=1.0,
                    description="Water flow velocity",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="m",
                    default=0.45,
                    description="Plant mortality rate",
                    min_value=0.01,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="Dn",
                    default=0.01,
                    description="Plant dispersal coefficient",
                    min_value=0.001,
                    max_value=1.0,
                ),
            ],
            num_fields=2,
            field_names=["w", "n"],
            reference="Klausmeier (1999) vegetation patterns",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        a = parameters.get("a", 2.0)
        v = parameters.get("v", 1.0)
        m = parameters.get("m", 0.45)
        Dn = parameters.get("Dn", 0.01)

        # Water equation with advection
        w_rhs = f"{a} - w - w * n**2"
        if v != 0:
            w_rhs += f" + {v} * d_dx(w)"

        return PDE(
            rhs={
                "w": w_rhs,
                "n": f"{Dn} * laplace(n) + w * n**2 - {m} * n",
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state with water and vegetation."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.1)

        # Uniform water, patchy vegetation
        w_data = 2.0 * np.ones(grid.shape)
        n_data = 0.5 + noise * np.random.randn(*grid.shape)
        n_data = np.clip(n_data, 0.01, 2.0)

        w = ScalarField(grid, w_data)
        w.label = "w"
        n = ScalarField(grid, n_data)
        n.label = "n"

        return FieldCollection([w, n])

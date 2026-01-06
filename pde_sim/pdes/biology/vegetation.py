"""Klausmeier vegetation model for semi-arid ecosystems."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("vegetation")
class VegetationPDE(MultiFieldPDEPreset):
    """Klausmeier vegetation model for semi-arid ecosystems.

    Standard formulation from visualpde.com:

        dw/dt = a - w - w*n² + v*∂w/∂x + Dw*∇²w
        dn/dt = w*n² - m*n + Dn*∇²n

    where:
        - w is water density
        - n is vegetation (plant) density
        - a is rainfall rate
        - v is water flow velocity (downhill advection)
        - m is plant mortality rate
        - Dw is water diffusion (spreading)
        - Dn is plant dispersal

    Produces banded vegetation patterns on hillslopes.

    Reference: Klausmeier (1999), visualpde.com
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="vegetation",
            category="biology",
            description="Klausmeier vegetation-water dynamics",
            equations={
                "w": "Dw * laplace(w) + a - w - w * n**2 + v * d_dx(w)",
                "n": "Dn * laplace(n) + w * n**2 - m * n",
            },
            parameters=[
                PDEParameter(
                    name="a",
                    default=2.0,
                    description="Rainfall rate",
                    min_value=0.1,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="v",
                    default=182.5,
                    description="Water flow velocity (downhill)",
                    min_value=0.0,
                    max_value=500.0,
                ),
                PDEParameter(
                    name="m",
                    default=0.45,
                    description="Plant mortality rate",
                    min_value=0.01,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="Dw",
                    default=1.0,
                    description="Water diffusion coefficient",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dn",
                    default=1.0,
                    description="Plant dispersal coefficient",
                    min_value=0.1,
                    max_value=10.0,
                ),
            ],
            num_fields=2,
            field_names=["w", "n"],
            reference="visualpde.com Klausmeier vegetation patterns",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        a = parameters.get("a", 2.0)
        v = parameters.get("v", 182.5)
        m = parameters.get("m", 0.45)
        Dw = parameters.get("Dw", 1.0)
        Dn = parameters.get("Dn", 1.0)

        # Water equation: diffusion + rainfall - loss - uptake by plants + advection
        w_rhs = f"{Dw} * laplace(w) + {a} - w - w * n**2"
        if v != 0:
            w_rhs += f" + {v} * d_dx(w)"

        # Vegetation equation: growth from water - mortality + dispersal
        n_rhs = f"{Dn} * laplace(n) + w * n**2 - {m} * n"

        return PDE(
            rhs={
                "w": w_rhs,
                "n": n_rhs,
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

"""Klausmeier vegetation model on terrain with topography."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("topography")
class TopographyPDE(MultiFieldPDEPreset):
    """Klausmeier vegetation on terrain with topography.

    Extended Klausmeier model from visualpde.com with terrain effects:

        dw/dt = a - w - w*n² + Dw*∇²w + V*∇·(w*∇T)
        dn/dt = w*n² - m*n + Dn*∇²n

    where T(x,y) is the terrain elevation (fixed field).
    The term V*∇·(w*∇T) models water flowing downhill along terrain gradients.

    For simplicity, we use T(x,y) = Tx*x + Ty*y (tilted plane),
    so ∇T = (Tx, Ty) is constant.

    Then ∇·(w*∇T) = ∇w·∇T = Tx*∂w/∂x + Ty*∂w/∂y

    Reference: visualpde.com hills and valleys
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="topography",
            category="biology",
            description="Klausmeier vegetation on terrain",
            equations={
                "w": "Dw*laplace(w) + a - w - w*n**2 + V*(Tx*d_dx(w) + Ty*d_dy(w))",
                "n": "Dn*laplace(n) + w*n**2 - m*n",
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
                PDEParameter(
                    name="V",
                    default=50.0,
                    description="Water transport strength on terrain",
                    min_value=0.0,
                    max_value=200.0,
                ),
                PDEParameter(
                    name="Tx",
                    default=0.1,
                    description="Terrain slope in x direction",
                    min_value=-1.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="Ty",
                    default=0.0,
                    description="Terrain slope in y direction",
                    min_value=-1.0,
                    max_value=1.0,
                ),
            ],
            num_fields=2,
            field_names=["w", "n"],
            reference="visualpde.com Klausmeier on topography",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        a = parameters.get("a", 2.0)
        m = parameters.get("m", 0.45)
        Dw = parameters.get("Dw", 1.0)
        Dn = parameters.get("Dn", 1.0)
        V = parameters.get("V", 50.0)
        Tx = parameters.get("Tx", 0.1)
        Ty = parameters.get("Ty", 0.0)

        # Water equation: diffusion + rainfall - loss - uptake + terrain transport
        w_rhs = f"{Dw} * laplace(w) + {a} - w - w * n**2"
        if V != 0 and (Tx != 0 or Ty != 0):
            w_rhs += f" + {V} * ({Tx} * d_dx(w) + {Ty} * d_dy(w))"

        # Vegetation equation
        n_rhs = f"{Dn} * laplace(n) + w * n**2 - {m} * n"

        return PDE(
            rhs={"w": w_rhs, "n": n_rhs},
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

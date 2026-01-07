"""Klausmeier model on topography with gravity-driven water flow."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("klausmeier-topography")
class KlausmeierTopographyPDE(MultiFieldPDEPreset):
    """Klausmeier model on realistic terrain.

    Extended vegetation-water model with topography-driven flow:

        dw/dt = a - w - w*n^2 + Dw*laplace(w) + V*div(w*grad(T))
        dn/dt = w*n^2 - m*n + Dn*laplace(n)

    where T(x,y) is the topographic height function.

    The advection term V*div(w*grad(T)) causes water to flow from high
    to low elevation, accumulating in valleys.

    Key phenomena:
        - Valley accumulation: water collects in low-lying areas
        - Hilltop stress: vegetation stress on exposed ridges
        - Pattern disruption: irregular terrain breaks regular stripes
        - Preferential colonization: vegetation in water-rich valleys

    Note: T is stored as a third field that remains constant after
    initial setup (with brief smoothing).

    References:
        Klausmeier, C. A. (1999). Science, 284(5421), 1826-1828.
        Saco et al. (2007). Hydrol. Earth Syst. Sci., 11(6), 1717-1730.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="klausmeier-topography",
            category="biology",
            description="Klausmeier vegetation on topography",
            equations={
                "n": "Dn * laplace(n) + w * n**2 - m * n",
                "w": "Dw * laplace(w) + a - w - w * n**2 + V * div(w * gradient(T))",
                "T": "0",  # Topography is static
            },
            parameters=[
                PDEParameter(
                    name="a",
                    default=1.1,
                    description="Rainfall rate",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="m",
                    default=0.63,
                    description="Plant mortality rate",
                    min_value=0.2,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="V",
                    default=100.0,
                    description="Gravity-driven flow strength",
                    min_value=0.0,
                    max_value=500.0,
                ),
                PDEParameter(
                    name="Dn",
                    default=1.0,
                    description="Plant diffusion coefficient",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dw",
                    default=2.0,
                    description="Water diffusion coefficient",
                    min_value=0.1,
                    max_value=10.0,
                ),
            ],
            num_fields=3,
            field_names=["n", "w", "T"],
            reference="Klausmeier (1999), Saco et al. (2007)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        a = parameters.get("a", 1.1)
        m = parameters.get("m", 0.63)
        V = parameters.get("V", 100.0)
        Dn = parameters.get("Dn", 1.0)
        Dw = parameters.get("Dw", 2.0)

        # The div(w*grad(T)) term implements topographic water flow
        # = w*laplace(T) + inner(gradient(w), gradient(T))
        return PDE(
            rhs={
                "n": f"{Dn} * laplace(n) + w * n**2 - {m} * n",
                "w": f"{Dw} * laplace(w) + {a} - w - w * n**2 + {V} * (w * laplace(T) + inner(gradient(w), gradient(T)))",
                "T": "0",  # Topography is static
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial state with topography."""
        a = ic_params.get("a", 1.1)
        noise = ic_params.get("noise", 0.1)

        np.random.seed(ic_params.get("seed"))

        # Initial water: near rainfall equilibrium with noise
        w_data = a * (1 + noise * np.random.randn(*grid.shape))
        w_data = np.maximum(w_data, 0.01)

        # Initial plants: small uniform with noise
        n_data = 0.1 + noise * np.random.randn(*grid.shape)
        n_data = np.maximum(n_data, 0.01)

        # Topography: default to simple gradient (can be overridden)
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Default: gentle slope in x direction
        slope = ic_params.get("slope", 0.1)
        T_data = slope * X

        n = ScalarField(grid, n_data)
        n.label = "n"
        w = ScalarField(grid, w_data)
        w.label = "w"
        T = ScalarField(grid, T_data)
        T.label = "T"

        return FieldCollection([n, w, T])

"""Bistable Allen-Cahn equation with Allee effects."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("bistable-allen-cahn")
class BistableAllenCahnPDE(ScalarPDEPreset):
    """Bistable Allen-Cahn equation modeling invasion with Allee effects.

    The bistable Allen-Cahn equation describes systems with two stable equilibria:

        du/dt = D * laplace(u) + u * (u - a) * (1 - u)

    where:
        - u is the population density (scaled to [0,1])
        - D is the diffusion coefficient
        - a is the Allee threshold parameter (0 < a < 1)

    Key properties:
        - u = 0 (extinction) and u = 1 (persistence) are stable
        - u = a is an unstable intermediate state
        - a < 0.5: Waves expand (u=1 invades u=0)
        - a > 0.5: Waves contract (u=0 invades u=1)
        - a = 0.5: Stationary waves

    Wave speed: c = sqrt(D/2) * (1 - 2a)

    References:
        Allen, S. M., & Cahn, J. W. (1979). Acta Metallurgica, 27(6), 1085-1095.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="bistable-allen-cahn",
            category="biology",
            description="Bistable Allen-Cahn equation with Allee effects",
            equations={
                "u": "D * laplace(u) + u * (u - a) * (1 - u)",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Diffusion coefficient",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="a",
                    default=0.5,
                    description="Allee threshold (0 < a < 1)",
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Allen & Cahn (1979)",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        a = parameters.get("a", 0.5)

        return PDE(
            rhs={"u": f"{D} * laplace(u) + u * (u - {a}) * (1 - u)"},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> ScalarField:
        """Create initial state for bistable Allen-Cahn.

        Supports:
        - gaussian-blobs: Localized population patches
        - step: Step function (left=0, right=1)
        - random-uniform: Random values in [0, 1]
        """
        seed = ic_params.get("seed")
        if seed is not None:
            np.random.seed(seed)

        if ic_type == "step":
            # Step function: 0 on left, 1 on right
            x_bounds = grid.axes_bounds[0]
            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(grid.axes_bounds[1][0], grid.axes_bounds[1][1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            mid = (x_bounds[0] + x_bounds[1]) / 2
            data = np.where(X < mid, 0.0, 1.0)
            return ScalarField(grid, data)

        # Default: use base class
        return super().create_initial_state(grid, ic_type, ic_params)

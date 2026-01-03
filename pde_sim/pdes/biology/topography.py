"""Population dynamics on varying terrain."""

from typing import Any

from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("topography")
class TopographyPDE(MultiFieldPDEPreset):
    """Population dynamics on varying terrain.

    Movement biased by terrain slope:

        du/dt = D * laplace(u) - alpha * div(u * grad(h)) + r * u * (1 - u)

    where h is a fixed elevation field affecting movement.
    Simplified: use advection toward low points.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="topography",
            category="biology",
            description="Population dynamics on terrain",
            equations={
                "u": "D * laplace(u) - alpha * (d_dx(u) * hx + d_dy(u) * hy) + r * u * (1 - u)",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.1,
                    description="Diffusion coefficient",
                    min_value=0.01,
                    max_value=0.5,
                ),
                PDEParameter(
                    name="alpha",
                    default=0.3,
                    description="Terrain influence strength",
                    min_value=0.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="r",
                    default=1.0,
                    description="Growth rate",
                    min_value=0.0,
                    max_value=3.0,
                ),
                PDEParameter(
                    name="hx",
                    default=0.1,
                    description="Terrain slope in x",
                    min_value=-1.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="hy",
                    default=0.0,
                    description="Terrain slope in y",
                    min_value=-1.0,
                    max_value=1.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Population on heterogeneous terrain",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 0.1)
        alpha = parameters.get("alpha", 0.3)
        r = parameters.get("r", 1.0)
        hx = parameters.get("hx", 0.1)
        hy = parameters.get("hy", 0.0)

        # Advection-diffusion with terrain
        rhs = f"{D} * laplace(u) - {alpha} * ({hx} * d_dx(u) + {hy} * d_dy(u)) + {r} * u * (1 - u)"

        return PDE(
            rhs={"u": rhs},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        return create_initial_condition(grid, ic_type, ic_params)

"""Fisher-KPP equation for population dynamics."""

from typing import Any

from pde import PDE, CartesianGrid

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("fisher-kpp")
class FisherKPPPDE(ScalarPDEPreset):
    """Fisher-KPP equation for population dynamics.

    The Fisher-KPP equation describes population growth with diffusion:

        du/dt = D * laplace(u) + r * u * (1 - u/K)

    where:
        - u is population density
        - D is diffusion coefficient
        - r is growth rate
        - K is carrying capacity

    Exhibits traveling wave solutions.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="fisher-kpp",
            category="biology",
            description="Fisher-KPP population dynamics",
            equations={
                "u": "D * laplace(u) + r * u * (1 - u / K)",
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
                    name="r",
                    default=1.0,
                    description="Growth rate",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="K",
                    default=1.0,
                    description="Carrying capacity",
                    min_value=0.1,
                    max_value=10.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Population invasion fronts",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        r = parameters.get("r", 1.0)
        K = parameters.get("K", 1.0)

        return PDE(
            rhs={"u": f"{D} * laplace(u) + {r} * u * (1 - u / {K})"},
            bc=self._convert_bc(bc),
        )

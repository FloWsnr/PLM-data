"""Fisher-KPP equation for population spread and traveling waves."""

from typing import Any

from pde import PDE, CartesianGrid

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("fisher-kpp")
class FisherKPPPDE(ScalarPDEPreset):
    """Fisher-KPP equation for population dynamics.

    The Fisher-KPP equation describes traveling waves of invasion:

        du/dt = D * laplace(u) + r * u * (1 - u/K)

    where:
        - u is the population density
        - D is the diffusion coefficient
        - r is the intrinsic growth rate
        - K is the carrying capacity

    Key result: Traveling wave solutions with minimum speed c = 2*sqrt(r*D).

    Applications:
        - Species invasion and range expansion
        - Epidemic spread (SIS model reduces to Fisher-KPP)
        - Tumor growth
        - Gene propagation

    References:
        Fisher, R. A. (1937). Annals of Eugenics, 7(4), 355-369.
        Kolmogorov, Petrovsky, Piskunov (1937). Moscow Univ. Math. Bull.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="fisher-kpp",
            category="biology",
            description="Fisher-KPP equation for population spread",
            equations={
                "u": "D * laplace(u) + r * u * (1 - u / K)",
            },
            parameters=[
                PDEParameter("D", "Diffusion coefficient"),
                PDEParameter("r", "Intrinsic growth rate"),
                PDEParameter("K", "Carrying capacity"),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Fisher (1937), Kolmogorov et al. (1937)",
            supported_dimensions=[1, 2, 3],
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

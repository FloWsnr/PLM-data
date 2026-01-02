"""Cahn-Hilliard equation for phase separation."""

from typing import Any

from pde import PDE, CartesianGrid

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("cahn-hilliard")
class CahnHilliardPDE(ScalarPDEPreset):
    """Cahn-Hilliard equation for phase separation.

    The Cahn-Hilliard equation describes phase separation in binary mixtures:

        du/dt = M * laplace(u³ - u - gamma * laplace(u))

    where:
        - u is the concentration difference between phases (-1 to 1)
        - M is mobility
        - gamma controls interface width

    The equation conserves the total amount of u.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="cahn-hilliard",
            category="physics",
            description="Cahn-Hilliard phase separation",
            equations={
                "u": "M * laplace(u**3 - u - gamma * laplace(u))",
            },
            parameters=[
                PDEParameter(
                    name="M",
                    default=1.0,
                    description="Mobility coefficient",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="gamma",
                    default=0.01,
                    description="Interface width parameter",
                    min_value=0.001,
                    max_value=1.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Phase separation in binary mixtures",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        M = parameters.get("M", 1.0)
        gamma = parameters.get("gamma", 0.01)

        return PDE(
            rhs={"u": f"{M} * laplace(u**3 - u - {gamma} * laplace(u))"},
            bc=self._convert_bc(bc),
        )

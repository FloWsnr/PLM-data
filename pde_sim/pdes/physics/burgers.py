"""Burgers' equation."""

from typing import Any

from pde import PDE, CartesianGrid

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("burgers")
class BurgersPDE(ScalarPDEPreset):
    """Burgers' equation.

    The viscous Burgers' equation describes shock wave formation:

        du/dt = nu * laplace(u) - u * d_dx(u)

    where nu is viscosity.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="burgers",
            category="physics",
            description="Burgers' equation (viscous)",
            equations={
                "u": "nu * laplace(u) - u * d_dx(u)",
            },
            parameters=[
                PDEParameter(
                    name="nu",
                    default=0.01,
                    description="Viscosity",
                    min_value=0.001,
                    max_value=1.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Shock wave formation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        nu = parameters.get("nu", 0.01)

        return PDE(
            rhs={"u": f"{nu} * laplace(u) - u * d_dx(u)"},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

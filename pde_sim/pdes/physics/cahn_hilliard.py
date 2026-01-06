"""Cahn-Hilliard equation for phase separation."""

from typing import Any

from pde import PDE, CartesianGrid

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("cahn-hilliard")
class CahnHilliardPDE(ScalarPDEPreset):
    """Cahn-Hilliard equation for phase separation with reaction term.

    Based on visualpde.com formulation:

        du/dt = r * laplace(u³ - u - g * laplace(u)) + u - u³

    where:
        - u is the concentration difference between phases (-1 to 1)
        - r controls the timescale of phase separation (mobility)
        - g controls interface width (surface tension)

    The reaction term (u - u³) drives phase separation while the
    Cahn-Hilliard term controls the interface dynamics.

    Reference: https://visualpde.com/nonlinear-physics/cahn-hilliard
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="cahn-hilliard",
            category="physics",
            description="Cahn-Hilliard phase separation with reaction",
            equations={
                "u": "r * laplace(u**3 - u - g * laplace(u)) + u - u**3",
            },
            parameters=[
                PDEParameter(
                    name="r",
                    default=0.01,
                    description="Phase separation timescale (mobility)",
                    min_value=0.001,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="g",
                    default=0.01,
                    description="Interface width parameter",
                    min_value=0.001,
                    max_value=0.1,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="https://visualpde.com/nonlinear-physics/cahn-hilliard",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        r = parameters.get("r", 0.01)
        g = parameters.get("g", 0.01)

        # Cahn-Hilliard with reaction: du/dt = r*laplace(u³-u-g*laplace(u)) + u - u³
        return PDE(
            rhs={"u": f"{r} * laplace(u**3 - u - {g} * laplace(u)) + u - u**3"},
            bc=self._convert_bc(bc),
        )

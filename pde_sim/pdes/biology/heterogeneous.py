"""Reaction-diffusion with spatially varying parameters."""

from typing import Any

from pde import PDE, CartesianGrid

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("heterogeneous")
class HeterogeneousPDE(ScalarPDEPreset):
    """Reaction-diffusion with spatially varying parameters.

    Fisher-KPP with space-dependent growth rate:

        du/dt = D * laplace(u) + r(x,y) * u * (1 - u)

    where r(x,y) varies sinusoidally across the domain,
    creating regions of favorable and unfavorable habitat.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="heterogeneous",
            category="biology",
            description="Reaction-diffusion with spatial heterogeneity",
            equations={"u": "D * laplace(u) + r(x,y) * u * (1 - u)"},
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.1,
                    description="Diffusion coefficient",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="r_mean",
                    default=1.0,
                    description="Mean growth rate",
                    min_value=0.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="r_amp",
                    default=0.5,
                    description="Growth rate amplitude",
                    min_value=0.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="freq",
                    default=2.0,
                    description="Spatial frequency of variation",
                    min_value=1.0,
                    max_value=10.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Heterogeneous environment model",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 0.1)
        r_mean = parameters.get("r_mean", 1.0)
        r_amp = parameters.get("r_amp", 0.5)
        freq = parameters.get("freq", 2.0)

        # r(x,y) = r_mean + r_amp * sin(freq * 2 * pi * x)
        # Use x coordinate directly in py-pde
        r_expr = f"({r_mean} + {r_amp} * sin({freq} * 2 * 3.14159 * x))"

        return PDE(
            rhs={"u": f"{D} * laplace(u) + {r_expr} * u * (1 - u)"},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

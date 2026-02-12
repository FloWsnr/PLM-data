"""FitzHugh-Nagumo model for excitable and pattern-forming systems."""

from typing import Any

from pde import PDE, CartesianGrid

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("fitzhugh-nagumo")
class FitzHughNagumoPDE(MultiFieldPDEPreset):
    """FitzHugh-Nagumo model for excitable media.

    A simplified model of neuronal action potentials:

        du/dt = laplace(u) + u - u^3 - v
        dv/dt = Dv * laplace(v) + e_v * (u - a_v*v - a_z)

    where:
        - u is the voltage variable (fast)
        - v is the recovery variable (slow)
        - Dv > 1 for pattern formation
        - e_v controls the timescale separation

    Key behaviors:
        - Excitability: threshold-triggered large excursions
        - Oscillations: sustained periodic behavior
        - Pattern formation: Turing-like patterns when Dv >> 1
        - Spiral waves: rotating spirals in excitable regime

    References:
        FitzHugh, R. (1961). Biophysical J., 1(6), 445-466.
        Nagumo, J. et al. (1962). Proc. IRE, 50(10), 2061-2070.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="fitzhugh-nagumo",
            category="biology",
            description="FitzHugh-Nagumo excitable media",
            equations={
                "u": "laplace(u) + u - u**3 - v",
                "v": "Dv * laplace(v) + e_v * (u - a_v * v - a_z)",
            },
            parameters=[
                PDEParameter("Dv", "Inhibitor diffusion coefficient"),
                PDEParameter("e_v", "Recovery timescale parameter"),
                PDEParameter("a_v", "Recovery slope coefficient"),
                PDEParameter("a_z", "Recovery offset parameter"),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="FitzHugh (1961), Nagumo et al. (1962)",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Dv = parameters.get("Dv", 20.0)
        e_v = parameters.get("e_v", 0.5)
        a_v = parameters.get("a_v", 1.0)
        a_z = parameters.get("a_z", -0.1)

        u_eq = "laplace(u) + u - u**3 - v"
        if Dv == 0:
            v_eq = f"{e_v} * (u - {a_v} * v - {a_z})"
        else:
            v_eq = f"{Dv} * laplace(v) + {e_v} * (u - {a_v} * v - {a_z})"

        return PDE(
            rhs={"u": u_eq, "v": v_eq},
            bc=self._convert_bc(bc),
        )


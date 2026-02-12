"""Three-species FitzHugh-Nagumo variant with competing oscillations and patterns."""

from typing import Any

from pde import PDE, CartesianGrid

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("fitzhugh-nagumo-3")
class FitzHughNagumo3PDE(MultiFieldPDEPreset):
    """Three-species FitzHugh-Nagumo variant.

    Extends the classic FitzHugh-Nagumo with a third species, creating
    competition between oscillations and pattern formation:

        du/dt = laplace(u) + u - u^3 - v
        dv/dt = Dv*laplace(v) + e_v*(u - a_v*v - a_w*w - a_z)
        dw/dt = Dw*laplace(w) + e_w*(u - w)

    where:
        - u is the voltage/activator (fast, with cubic nonlinearity)
        - v is the first recovery variable (medium timescale)
        - w is the second recovery variable (slow, pattern-forming)
        - Dv, Dw > 1 for pattern formation

    Key behaviors:
        - Low a_v (< 0.3): Patterns formed initially are destroyed by oscillations
        - High a_v (>= 0.3): Patterns stabilize and overtake oscillations
        - Competition between local oscillations and spatial patterns

    Reference:
        https://visualpde.com/nonlinear-dynamics/fitzhugh-nagumo-3
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="fitzhugh-nagumo-3",
            category="biology",
            description="Three-species FitzHugh-Nagumo with pattern-oscillation competition",
            equations={
                "u": "laplace(u) + u - u**3 - v",
                "v": "Dv * laplace(v) + e_v * (u - a_v*v - a_w*w - a_z)",
                "w": "Dw * laplace(w) + e_w * (u - w)",
            },
            parameters=[
                PDEParameter("Dv", "Diffusion for v (first recovery)"),
                PDEParameter("Dw", "Diffusion for w (second recovery, pattern-forming)"),
                PDEParameter("a_v", "Self-inhibition of v (controls pattern vs oscillation)"),
                PDEParameter("e_v", "Timescale parameter for v"),
                PDEParameter("e_w", "Timescale parameter for w"),
                PDEParameter("a_w", "Coupling coefficient v-w"),
                PDEParameter("a_z", "Offset parameter for v dynamics"),
            ],
            num_fields=3,
            field_names=["u", "v", "w"],
            reference="VisualPDE FitzHugh-Nagumo-3",
            supported_dimensions=[2],  # Currently 2D only (IC uses 2D meshgrid)
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Dv = parameters.get("Dv", 40.0)
        Dw = parameters.get("Dw", 200.0)
        a_v = parameters.get("a_v", 0.2)
        e_v = parameters.get("e_v", 0.2)
        e_w = parameters.get("e_w", 1.0)
        a_w = parameters.get("a_w", 0.5)
        a_z = parameters.get("a_z", -0.1)

        # u equation: cubic activator dynamics
        u_eq = "laplace(u) + u - u**3 - v"

        # v equation: first recovery with coupling to w
        v_eq = f"{Dv} * laplace(v) + {e_v} * (u - {a_v}*v - {a_w}*w - {a_z})"

        # w equation: second recovery (slow, pattern-forming)
        w_eq = f"{Dw} * laplace(w) + {e_w} * (u - w)"

        return PDE(
            rhs={"u": u_eq, "v": v_eq, "w": w_eq},
            bc=self._convert_bc(bc),
        )


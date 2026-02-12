"""Hyperbolic Brusselator for Turing wave instabilities."""

from typing import Any

from pde import PDE, CartesianGrid

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("hyperbolic-brusselator")
class HyperbolicBrusselatorPDE(MultiFieldPDEPreset):
    """Hyperbolic Brusselator with Turing wave instability.

    A modified reaction-diffusion system with inertial effects that exhibits
    oscillating spatial patterns - Turing instabilities with complex eigenvalues.

    The hyperbolic Brusselator introduces a relaxation time (tau):

        tau * d2u/dt2 + du/dt = Du * laplace(u) + a - (b+1)*u + u^2*v
        tau * d2v/dt2 + dv/dt = Dv * laplace(v) + b*u - u^2*v

    Implemented as a first-order system with auxiliary variables (w, q):

        du/dt = w
        dv/dt = q
        dw/dt = (Du/tau)*laplace(u) + eps*laplace(w) + (a + u^2*v - (b+1)*u - w)/tau
        dq/dt = (Dv/tau)*laplace(v) + eps*laplace(q) + (b*u - u^2*v - q)/tau

    Key features:
        - tau > 0: Memory/inertia effects create Turing wave instabilities
        - Spatial modes oscillate in time (complex eigenvalues)
        - Can produce traveling or standing wave patterns

    References:
        Zemskov et al. (2022). arXiv:2204.13820
        Fort & Mendez (2002). Phys. Rev. Lett. 89:178101
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="hyperbolic-brusselator",
            category="biology",
            description="Hyperbolic Brusselator with Turing wave instability",
            equations={
                "u": "w",
                "v": "q",
                "w": "(Du/tau)*laplace(u) + eps*laplace(w) + (a + u^2*v - (b+1)*u - w)/tau",
                "q": "(Dv/tau)*laplace(v) + eps*laplace(q) + (b*u - u^2*v - q)/tau",
            },
            parameters=[
                PDEParameter("tau", "Relaxation time (memory parameter)"),
                PDEParameter("Du", "Activator diffusion coefficient"),
                PDEParameter("Dv", "Inhibitor diffusion coefficient"),
                PDEParameter("a", "Brusselator feed rate"),
                PDEParameter("b", "Brusselator removal rate"),
                PDEParameter("eps", "Numerical diffusion for velocity variables"),
            ],
            num_fields=4,
            field_names=["u", "v", "w", "q"],
            reference="Zemskov et al. (2022)",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        tau = parameters.get("tau", 1.0)
        Du = parameters.get("Du", 2.0)
        Dv = parameters.get("Dv", 1.0)
        a = parameters.get("a", 5.0)
        b = parameters.get("b", 9.0)
        eps = parameters.get("eps", 0.001)

        # Avoid division by zero
        if tau <= 0:
            tau = 0.001

        return PDE(
            rhs={
                "u": "w",
                "v": "q",
                "w": f"{Du/tau} * laplace(u) + {eps} * laplace(w) + ({a} + u**2 * v - ({b} + 1) * u - w) / {tau}",
                "q": f"{Dv/tau} * laplace(v) + {eps} * laplace(q) + ({b} * u - u**2 * v - q) / {tau}",
            },
            bc=self._convert_bc(bc),
        )


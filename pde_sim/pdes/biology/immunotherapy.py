"""Immunotherapy model for tumor-immune interactions."""

from typing import Any

from pde import PDE, CartesianGrid

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("immunotherapy")
class ImmunotherapyPDE(MultiFieldPDEPreset):
    """Three-species tumor-immune-cytokine model.

    Describes spatial pattern formation in cancer immunotherapy:

        du/dt = Du*laplace(u) + c*v - mu*u + p_u*u*w/(1+w) + s_u
        dv/dt = Dv*laplace(v) + v*(1-v) - p_v*u*v/(g_v+v)
        dw/dt = Dw*laplace(w) + p_w*u*v/(g_w+v) - nu*w + s_w

    where:
        - u is the effector cell density (immune)
        - v is the tumor cell density
        - w is the cytokine concentration (IL-2)

    Key insight: Turing patterns in tumors can confer treatment resistance.
    The tumor diffusion (Dv) is typically much slower than effector/cytokine diffusion.

    References:
        Kuznetsov et al. (1994). Bull. Math. Biol., 56(2), 295-321.
        Kirschner & Panetta (1998). J. Math. Biol., 37(3), 235-252.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="immunotherapy",
            category="biology",
            description="Tumor-immune-cytokine immunotherapy model",
            equations={
                "u": "Du*laplace(u) + c*v - mu*u + p_u*u*w/(1+w) + s_u",
                "v": "Dv*laplace(v) + v*(1-v) - p_v*u*v/(g_v+v)",
                "w": "Dw*laplace(w) + p_w*u*v/(g_w+v) - nu*w + s_w",
            },
            parameters=[
                PDEParameter("Du", "Effector cell diffusion"),
                PDEParameter("Dv", "Tumor diffusion (slow)"),
                PDEParameter("Dw", "Cytokine diffusion"),
                PDEParameter("c", "Tumor-induced effector recruitment (alpha)"),
                PDEParameter("mu", "Effector decay rate (mu_u)"),
                PDEParameter("p_u", "Effector proliferation rate (rho_u)"),
                PDEParameter("g_v", "Tumor saturation constant (gamma_v)"),
                PDEParameter("p_v", "Tumor killing rate coefficient"),
                PDEParameter("p_w", "Cytokine production rate (rho_w)"),
                PDEParameter("g_w", "Cytokine saturation constant (gamma_w)"),
                PDEParameter("nu", "Cytokine decay rate (mu_w)"),
                PDEParameter("s_u", "Constant effector source (sigma_u)"),
                PDEParameter("s_w", "Constant cytokine source (sigma_w)"),
            ],
            num_fields=3,
            field_names=["u", "v", "w"],
            reference="Kuznetsov et al. (1994)",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Du = parameters.get("Du", 0.5)
        Dv = parameters.get("Dv", 0.008)
        Dw = parameters.get("Dw", 4.0)
        c = parameters.get("c", 0.3)
        mu = parameters.get("mu", 0.167)
        p_u = parameters.get("p_u", 0.69167)
        g_v = parameters.get("g_v", 0.1)
        p_v = parameters.get("p_v", 1.0)
        p_w = parameters.get("p_w", 27.778)
        g_w = parameters.get("g_w", 0.001)
        nu = parameters.get("nu", 55.55556)
        s_u = parameters.get("s_u", 0.0)
        s_w = parameters.get("s_w", 10.0)

        u_rhs = f"{Du} * laplace(u) + {c} * v - {mu} * u + {p_u} * u * w / (1 + w) + {s_u}"
        v_rhs = f"{Dv} * laplace(v) + v * (1 - v) - {p_v} * u * v / ({g_v} + v)"
        w_rhs = f"{Dw} * laplace(w) + {p_w} * u * v / ({g_w} + v) - {nu} * w + {s_w}"

        return PDE(
            rhs={"u": u_rhs, "v": v_rhs, "w": w_rhs},
            bc=self._convert_bc(bc),
        )


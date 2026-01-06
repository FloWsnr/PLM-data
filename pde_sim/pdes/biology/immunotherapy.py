"""Immunotherapy model for tumor-immune interactions."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("immunotherapy")
class ImmunotherapyPDE(MultiFieldPDEPreset):
    """Three-species tumor-immune-cytokine model.

    Describes spatial pattern formation in cancer immunotherapy:

        du/dt = Du*laplace(u) + c*v - mu*u + p_u*u*w/(1+w) + s_u
        dv/dt = laplace(v) + v*(1-v) - u*v/(g_v+v)
        dw/dt = Dw*laplace(w) + p_w*u*v/(g_w+v) - nu*w + s_w

    where:
        - u is the effector cell density (immune)
        - v is the tumor cell density
        - w is the cytokine concentration (IL-2)

    Key insight: Turing patterns in tumors can confer treatment resistance.

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
                "v": "laplace(v) + v*(1-v) - u*v/(g_v+v)",
                "w": "Dw*laplace(w) + p_w*u*v/(g_w+v) - nu*w + s_w",
            },
            parameters=[
                PDEParameter(
                    name="Du",
                    default=0.5,
                    description="Effector cell diffusion",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dw",
                    default=4.0,
                    description="Cytokine diffusion",
                    min_value=0.1,
                    max_value=20.0,
                ),
                PDEParameter(
                    name="c",
                    default=0.3,
                    description="Tumor-induced effector recruitment (alpha)",
                    min_value=0.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="mu",
                    default=0.167,
                    description="Effector decay rate (mu_u)",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="p_u",
                    default=0.69167,
                    description="Effector proliferation rate (rho_u)",
                    min_value=0.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="g_v",
                    default=0.1,
                    description="Tumor saturation constant (gamma_v)",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="p_w",
                    default=27.778,
                    description="Cytokine production rate (rho_w)",
                    min_value=0.0,
                    max_value=50.0,
                ),
                PDEParameter(
                    name="g_w",
                    default=0.001,
                    description="Cytokine saturation constant (gamma_w)",
                    min_value=0.0001,
                    max_value=0.1,
                ),
                PDEParameter(
                    name="nu",
                    default=55.55556,
                    description="Cytokine decay rate (mu_w)",
                    min_value=1.0,
                    max_value=100.0,
                ),
                PDEParameter(
                    name="s_u",
                    default=0.0,
                    description="Constant effector source (sigma_u)",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="s_w",
                    default=10.0,
                    description="Constant cytokine source (sigma_w)",
                    min_value=0.0,
                    max_value=50.0,
                ),
            ],
            num_fields=3,
            field_names=["u", "v", "w"],
            reference="Kuznetsov et al. (1994)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Du = parameters.get("Du", 0.5)
        Dw = parameters.get("Dw", 4.0)
        c = parameters.get("c", 0.3)
        mu = parameters.get("mu", 0.167)
        p_u = parameters.get("p_u", 0.69167)
        g_v = parameters.get("g_v", 0.1)
        p_w = parameters.get("p_w", 27.778)
        g_w = parameters.get("g_w", 0.001)
        nu = parameters.get("nu", 55.55556)
        s_u = parameters.get("s_u", 0.0)
        s_w = parameters.get("s_w", 10.0)

        u_rhs = f"{Du} * laplace(u) + {c} * v - {mu} * u + {p_u} * u * w / (1 + w) + {s_u}"
        v_rhs = f"laplace(v) + v * (1 - v) - u * v / ({g_v} + v)"
        w_rhs = f"{Dw} * laplace(w) + {p_w} * u * v / ({g_w} + v) - {nu} * w + {s_w}"

        return PDE(
            rhs={"u": u_rhs, "v": v_rhs, "w": w_rhs},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial tumor with immune cells and cytokine."""
        noise = ic_params.get("noise", 0.01)
        np.random.seed(ic_params.get("seed"))

        # Small tumor in center
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        cx, cy = (x_bounds[0] + x_bounds[1]) / 2, (y_bounds[0] + y_bounds[1]) / 2
        Lx = x_bounds[1] - x_bounds[0]
        r_sq = ((X - cx) / Lx) ** 2 + ((Y - cy) / Lx) ** 2

        # Tumor: localized in center
        v_data = 0.8 * np.exp(-r_sq / 0.01) + noise * np.random.randn(*grid.shape)
        v_data = np.clip(v_data, 0.0, 1.0)

        # Effector cells: low uniform + noise
        u_data = 0.1 + noise * np.random.randn(*grid.shape)
        u_data = np.maximum(u_data, 0.0)

        # Cytokine: low uniform
        w_data = 0.1 + noise * np.random.randn(*grid.shape)
        w_data = np.maximum(w_data, 0.0)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"
        w = ScalarField(grid, w_data)
        w.label = "w"

        return FieldCollection([u, v, w])

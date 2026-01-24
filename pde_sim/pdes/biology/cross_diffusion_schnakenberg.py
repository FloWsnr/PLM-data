"""Cross-Diffusion Schnakenberg model for dark solitons and extended Turing patterns."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("cross-diffusion-schnakenberg")
class CrossDiffusionSchnakenbergPDE(MultiFieldPDEPreset):
    """Cross-Diffusion Schnakenberg system.

    Extended Schnakenberg with cross-diffusion terms:

        du/dt = div(Duu * grad(u) + Duv * grad(v)) + a - u + u^2 * v
        dv/dt = div(Dvu * grad(u) + Dvv * grad(v)) + b - u^2 * v

    Key features:
        - Equal self-diffusion (Duu = Dvv = 1)
        - Asymmetric cross-diffusion (Duv = 3, Dvu = 0.2)
        - Enables pattern formation without differential self-diffusion
        - Produces dark solitons (localized inverted spots)

    The cross-diffusion terms represent:
        - Duv: Flux of u driven by gradients in v
        - Dvu: Flux of v driven by gradients in u

    References:
        Schnakenberg, J. (1979). J. Theor. Biol., 81(3), 389-400.
        Vanag & Epstein (2009). PCCP, 11(6), 897-912.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="cross-diffusion-schnakenberg",
            category="biology",
            description="Schnakenberg with cross-diffusion for dark solitons",
            equations={
                "u": "Duu * laplace(u) + Duv * laplace(v) + a - u + u**2 * v",
                "v": "Dvu * laplace(u) + Dvv * laplace(v) + b - u**2 * v",
            },
            parameters=[
                PDEParameter("Duu", "Self-diffusion of u"),
                PDEParameter("Duv", "Cross-diffusion u-v"),
                PDEParameter("Dvu", "Cross-diffusion v-u"),
                PDEParameter("Dvv", "Self-diffusion of v"),
                PDEParameter("a", "Production rate parameter"),
                PDEParameter("b", "Production rate parameter"),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Vanag & Epstein (2009)",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Duu = parameters.get("Duu", 1.0)
        Duv = parameters.get("Duv", 3.0)
        Dvu = parameters.get("Dvu", 0.2)
        Dvv = parameters.get("Dvv", 1.0)
        a = parameters.get("a", 0.01)
        b = parameters.get("b", 2.5)

        return PDE(
            rhs={
                "u": f"{Duu} * laplace(u) + {Duv} * laplace(v) + {a} - u + u**2 * v",
                "v": f"{Dvu} * laplace(u) + {Dvv} * laplace(v) + {b} - u**2 * v",
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial state near the homogeneous steady state with perturbation."""
        a = ic_params.get("a", 0.01)
        b = ic_params.get("b", 2.5)
        noise = ic_params.get("noise", 0.01)

        u_ss = a + b
        v_ss = b / (u_ss ** 2)

        np.random.seed(ic_params.get("seed"))
        u_data = u_ss + noise * np.random.randn(*grid.shape)
        v_data = v_ss + noise * np.random.randn(*grid.shape)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

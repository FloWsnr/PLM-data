"""Cross-diffusion Schnakenberg system."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("cross-diffusion")
class CrossDiffusionPDE(MultiFieldPDEPreset):
    """Cross-diffusion Schnakenberg system.

    Schnakenberg kinetics with cross-diffusion from visualpde.com:

        du/dt = ∇·(Duu*∇u + Duv*∇v) + a - u + u²v
        dv/dt = ∇·(Dvu*∇u + Dvv*∇v) + b - u²v

    This extends the standard Schnakenberg model with cross-diffusion terms
    that allow pattern formation even when Duu = Dvv (equal self-diffusion).

    Cross-diffusion can produce localized 'dark soliton' patterns and
    extends the parameter space for Turing instabilities.

    Reference: visualpde.com cross-diffusion
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="cross-diffusion",
            category="biology",
            description="Cross-diffusion Schnakenberg pattern formation",
            equations={
                "u": "div(Duu*gradient(u) + Duv*gradient(v)) + a - u + u**2*v",
                "v": "div(Dvu*gradient(u) + Dvv*gradient(v)) + b - u**2*v",
            },
            parameters=[
                PDEParameter(
                    name="Duu",
                    default=1.0,
                    description="Self-diffusion of u",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Duv",
                    default=0.0,
                    description="Cross-diffusion: u responds to v gradient",
                    min_value=-10.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dvu",
                    default=-10.0,
                    description="Cross-diffusion: v responds to u gradient",
                    min_value=-20.0,
                    max_value=20.0,
                ),
                PDEParameter(
                    name="Dvv",
                    default=1.0,
                    description="Self-diffusion of v",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="a",
                    default=0.1,
                    description="Feed rate for u",
                    min_value=0.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="b",
                    default=0.9,
                    description="Feed rate for v",
                    min_value=0.1,
                    max_value=2.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="visualpde.com cross-diffusion Schnakenberg",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Duu = parameters.get("Duu", 1.0)
        Duv = parameters.get("Duv", 0.0)
        Dvu = parameters.get("Dvu", -10.0)
        Dvv = parameters.get("Dvv", 1.0)
        a = parameters.get("a", 0.1)
        b = parameters.get("b", 0.9)

        # Cross-diffusion: ∇·(D_ij ∇f_j) = D_ij * laplace(f_j) for constant D
        u_rhs = f"{Duu} * laplace(u) + {Duv} * laplace(v) + {a} - u + u**2 * v"
        v_rhs = f"{Dvu} * laplace(u) + {Dvv} * laplace(v) + {b} - u**2 * v"

        return PDE(
            rhs={"u": u_rhs, "v": v_rhs},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state for cross-diffusion system."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.1)

        u_data = 0.5 + noise * np.random.randn(*grid.shape)
        v_data = 0.5 + noise * np.random.randn(*grid.shape)
        u_data = np.clip(u_data, 0.01, 1.0)
        v_data = np.clip(v_data, 0.01, 1.0)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

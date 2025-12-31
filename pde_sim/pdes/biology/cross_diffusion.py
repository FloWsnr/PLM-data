"""Cross-diffusion system (Shigesada-Kawasaki-Teramoto model)."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("cross-diffusion")
class CrossDiffusionPDE(MultiFieldPDEPreset):
    """Cross-diffusion system (Shigesada-Kawasaki-Teramoto model).

    Two species with density-dependent diffusion:

        du/dt = laplace((d1 + a11*u + a12*v) * u) + u * (r1 - b1*u - c1*v)
        dv/dt = laplace((d2 + a21*u + a22*v) * v) + v * (r2 - b2*v - c2*u)

    Simplified version:
        du/dt = D1 * laplace(u) + alpha * laplace(u*v) + u * (1 - u - v)
        dv/dt = D2 * laplace(v) + beta * laplace(u*v) + v * (1 - u - v)

    Cross-diffusion can produce patterns even without Turing instability.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="cross-diffusion",
            category="biology",
            description="Cross-diffusion pattern formation",
            equations={
                "u": "D1 * laplace(u) + alpha * laplace(u*v) + u * (1 - u - v)",
                "v": "D2 * laplace(v) + beta * laplace(u*v) + v * (1 - u - v)",
            },
            parameters=[
                PDEParameter(
                    name="D1",
                    default=0.1,
                    description="Diffusion of species u",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="D2",
                    default=0.1,
                    description="Diffusion of species v",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="alpha",
                    default=1.0,
                    description="Cross-diffusion coefficient for u",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="beta",
                    default=0.0,
                    description="Cross-diffusion coefficient for v",
                    min_value=0.0,
                    max_value=10.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Shigesada-Kawasaki-Teramoto cross-diffusion",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D1 = parameters.get("D1", 0.1)
        D2 = parameters.get("D2", 0.1)
        alpha = parameters.get("alpha", 1.0)
        beta = parameters.get("beta", 0.0)

        u_rhs = f"{D1} * laplace(u) + {alpha} * laplace(u*v) + u * (1 - u - v)"
        v_rhs = f"{D2} * laplace(v) + {beta} * laplace(u*v) + v * (1 - u - v)"

        return PDE(
            rhs={"u": u_rhs, "v": v_rhs},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
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

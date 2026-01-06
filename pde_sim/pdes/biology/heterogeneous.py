"""Heterogeneous Gierer-Meinhardt system with spatial variation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("heterogeneous")
class HeterogeneousPDE(MultiFieldPDEPreset):
    """Heterogeneous Gierer-Meinhardt system.

    Gierer-Meinhardt model with spatial heterogeneity from visualpde.com:

        du/dt = ∇²u + a + G(x) + u²/v - [b + H(x)]*u
        dv/dt = D∇²v + u² - c*v

    where G(x) and H(x) are spatial variation functions:
        G(x) = A*x/L  (linear variation in production)
        H(x) = B*x/L  (linear variation in decay)

    This creates spatially varying patterns, spike movement, and
    interesting boundary-induced dynamics.

    Reference: visualpde.com heterogeneous dynamics
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="heterogeneous",
            category="biology",
            description="Heterogeneous Gierer-Meinhardt pattern formation",
            equations={
                "u": "laplace(u) + a + G(x) + u**2/v - (b + H(x))*u",
                "v": "D * laplace(v) + u**2 - c*v",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=40.0,
                    description="Inhibitor diffusion ratio",
                    min_value=1.0,
                    max_value=200.0,
                ),
                PDEParameter(
                    name="a",
                    default=0.1,
                    description="Basal activator production",
                    min_value=0.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="b",
                    default=1.0,
                    description="Activator decay rate",
                    min_value=0.1,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="c",
                    default=1.0,
                    description="Inhibitor decay rate",
                    min_value=0.1,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="A",
                    default=0.0,
                    description="Spatial variation in production G(x)=A*x/L",
                    min_value=-2.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="B",
                    default=0.0,
                    description="Spatial variation in decay H(x)=B*x/L",
                    min_value=-10.0,
                    max_value=10.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="visualpde.com heterogeneous dynamics",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 40.0)
        a = parameters.get("a", 0.1)
        b = parameters.get("b", 1.0)
        c = parameters.get("c", 1.0)
        A = parameters.get("A", 0.0)
        B = parameters.get("B", 0.0)

        # G(x) = A*x (assuming domain [0,1] so L=1)
        # H(x) = B*x
        G_expr = f"{A} * x" if A != 0 else "0"
        H_expr = f"{B} * x" if B != 0 else "0"

        u_rhs = f"laplace(u) + {a} + ({G_expr}) + u**2 / (v + 1e-10) - ({b} + ({H_expr})) * u"
        v_rhs = f"{D} * laplace(v) + u**2 - {c} * v"

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
        """Create initial state near steady state with perturbation."""
        noise = ic_params.get("noise", 0.01)
        np.random.seed(ic_params.get("seed"))

        # Initial values near steady state
        u_data = 1.0 + noise * np.random.randn(*grid.shape)
        v_data = 1.0 + noise * np.random.randn(*grid.shape)

        # Ensure positive values
        u_data = np.maximum(u_data, 0.01)
        v_data = np.maximum(v_data, 0.01)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

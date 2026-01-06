"""Gierer-Meinhardt activator-inhibitor system for morphogenesis."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("gierer-meinhardt")
class GiererMeinhardtPDE(MultiFieldPDEPreset):
    """Gierer-Meinhardt activator-inhibitor system.

    A classical model for biological pattern formation:

        du/dt = laplace(u) + a + u^2/v - b*u
        dv/dt = D * laplace(v) + u^2 - c*v

    where:
        - u is the activator concentration
        - v is the inhibitor concentration
        - D > 1 is required for pattern formation
        - a is basal activator production
        - b is activator decay rate
        - c is inhibitor decay rate

    Key phenomena:
        - Spot patterns: default behavior
        - Stripe instability: stripes break into spots
        - Labyrinthine patterns: with saturation term

    References:
        Gierer & Meinhardt (1972). Kybernetik, 12(1), 30-39.
        Meinhardt & Gierer (2000). BioEssays, 22(8), 753-760.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="gierer-meinhardt",
            category="biology",
            description="Gierer-Meinhardt activator-inhibitor pattern formation",
            equations={
                "u": "laplace(u) + a + u**2 / v - b * u",
                "v": "D * laplace(v) + u**2 - c * v",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=100.0,
                    description="Inhibitor diffusion ratio (D > 1 required)",
                    min_value=1.0,
                    max_value=500.0,
                ),
                PDEParameter(
                    name="a",
                    default=0.5,
                    description="Basal activator production",
                    min_value=0.0,
                    max_value=5.0,
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
                    default=6.1,
                    description="Inhibitor decay rate",
                    min_value=0.1,
                    max_value=20.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Gierer & Meinhardt (1972)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 100.0)
        a = parameters.get("a", 0.5)
        b = parameters.get("b", 1.0)
        c = parameters.get("c", 6.1)

        # Add small epsilon to prevent division by zero
        u_rhs = f"laplace(u) + {a} + u**2 / (v + 1e-10) - {b} * u"
        v_rhs = f"{D} * laplace(v) + u**2 - {c} * v"

        return PDE(
            rhs={
                "u": u_rhs,
                "v": v_rhs,
            },
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

        # Initial values near steady state (approximate)
        u0 = ic_params.get("u0", 1.0)
        v0 = ic_params.get("v0", 1.0)

        np.random.seed(ic_params.get("seed"))
        u_data = u0 * (1 + noise * np.random.randn(*grid.shape))
        v_data = v0 * (1 + noise * np.random.randn(*grid.shape))

        # Ensure positive values
        u_data = np.maximum(u_data, 0.01)
        v_data = np.maximum(v_data, 0.01)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

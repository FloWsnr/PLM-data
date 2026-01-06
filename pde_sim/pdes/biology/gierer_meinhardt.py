"""Gierer-Meinhardt activator-inhibitor system."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("gierer-meinhardt")
class GiererMeinhardtPDE(MultiFieldPDEPreset):
    """Gierer-Meinhardt activator-inhibitor system.

    Standard formulation from visualpde.com:

        du/dt = Du * laplace(u) + a + u²/v - b*u
        dv/dt = Dv * laplace(v) + u² - c*v

    where:
        - u is the activator concentration
        - v is the inhibitor concentration
        - Du, Dv are diffusion coefficients (Dv >> Du for patterns)
        - a is basal activator production
        - b is activator decay rate
        - c is inhibitor decay rate

    For Turing patterns: Dv/Du >> 1 (D > 1, typically ~10-100)

    Reference: Gierer & Meinhardt (1972), visualpde.com
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="gierer-meinhardt",
            category="biology",
            description="Gierer-Meinhardt activator-inhibitor pattern formation",
            equations={
                "u": "Du * laplace(u) + a + u**2 / v - b * u",
                "v": "Dv * laplace(v) + u**2 - c * v",
            },
            parameters=[
                PDEParameter(
                    name="Du",
                    default=1.0,
                    description="Activator diffusion coefficient",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dv",
                    default=40.0,
                    description="Inhibitor diffusion coefficient (D = Dv/Du > 1)",
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
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="visualpde.com Gierer-Meinhardt pattern formation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Du = parameters.get("Du", 1.0)
        Dv = parameters.get("Dv", 40.0)
        a = parameters.get("a", 0.1)
        b = parameters.get("b", 1.0)
        c = parameters.get("c", 1.0)

        # Build equation strings (add small epsilon to prevent division by zero)
        u_rhs = f"{Du} * laplace(u) + {a} + u**2 / (v + 1e-10) - {b} * u"
        v_rhs = f"{Dv} * laplace(v) + u**2 - {c} * v"

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

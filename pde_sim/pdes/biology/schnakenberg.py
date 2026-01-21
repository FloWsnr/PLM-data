"""Schnakenberg reaction-diffusion system for Turing patterns."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("schnakenberg")
class SchnakenbergPDE(MultiFieldPDEPreset):
    """Schnakenberg reaction-diffusion system.

    The Schnakenberg model produces Turing patterns:

        du/dt = laplace(u) + a - u + u^2 * v
        dv/dt = Dv * laplace(v) + b - u^2 * v

    where:
        - u is the activator concentration
        - v is the inhibitor concentration
        - Dv > 1 is required for pattern formation

    Key phenomena:
        - Spot patterns: Form when Dv is large (e.g., Dv = 100)
        - Stripe patterns: Form when Dv is reduced (e.g., Dv = 30)

    References:
        Schnakenberg, J. (1979). J. Theor. Biol., 81(3), 389-400.
        Turing, A. M. (1952). Phil. Trans. R. Soc. B, 237(641), 37-72.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="schnakenberg",
            category="biology",
            description="Schnakenberg Turing pattern system",
            equations={
                "u": "laplace(u) + a - u + u**2 * v",
                "v": "Dv * laplace(v) + b - u**2 * v",
            },
            parameters=[
                PDEParameter(
                    name="Dv",
                    default=100.0,
                    description="Inhibitor diffusion coefficient",
                    min_value=0.0,
                    max_value=100.0,
                ),
                PDEParameter(
                    name="a",
                    default=0.01,
                    description="Production rate parameter",
                    min_value=0.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="b",
                    default=2.0,
                    description="Production rate parameter",
                    min_value=0.0,
                    max_value=5.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Schnakenberg (1979)",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Dv = parameters.get("Dv", 100.0)
        a = parameters.get("a", 0.01)
        b = parameters.get("b", 2.0)

        return PDE(
            rhs={
                "u": f"laplace(u) + {a} - u + u**2 * v",
                "v": f"{Dv} * laplace(v) + {b} - u**2 * v",
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
        # Steady state: u* = a + b, v* = b / (a + b)^2
        a = ic_params.get("a", 0.01)
        b = ic_params.get("b", 2.0)
        noise = ic_params.get("noise", 0.01)

        u_ss = a + b
        v_ss = b / (u_ss ** 2)

        # Add small random perturbation
        np.random.seed(ic_params.get("seed"))
        u_data = u_ss + noise * np.random.randn(*grid.shape)
        v_data = v_ss + noise * np.random.randn(*grid.shape)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

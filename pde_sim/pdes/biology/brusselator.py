"""Brusselator reaction-diffusion system for oscillating chemical patterns."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("brusselator")
class BrusselatorPDE(MultiFieldPDEPreset):
    """Brusselator reaction-diffusion system.

    The Brusselator describes oscillating chemical reactions:

        du/dt = laplace(u) + a - (b+1)*u + u^2*v
        dv/dt = Dv * laplace(v) + b*u - u^2*v

    where:
        - u is the activator concentration
        - v is the inhibitor concentration
        - a, b > 0 are kinetic parameters
        - Dv is the diffusion coefficient ratio

    Stability conditions:
        - Homogeneous equilibrium stable for b - 1 < a^2
        - Turing instability for Dv > a^2 / (sqrt(b) - 1)^2
        - For a = 2, b = 3: critical Dv ~ 7.46

    References:
        Prigogine & Lefever (1968). J. Chem. Phys., 48(4), 1695-1700.
        Nicolis & Prigogine (1977). Self-Organization in Nonequilibrium Systems.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="brusselator",
            category="biology",
            description="Brusselator oscillating reaction-diffusion",
            equations={
                "u": "laplace(u) + a - (b + 1) * u + u**2 * v",
                "v": "Dv * laplace(v) + b * u - u**2 * v",
            },
            parameters=[
                PDEParameter("Dv", "Inhibitor diffusion coefficient"),
                PDEParameter("a", "Kinetic parameter a"),
                PDEParameter("b", "Kinetic parameter b"),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Prigogine & Lefever (1968)",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Dv = parameters.get("Dv", 8.0)
        a = parameters.get("a", 2.0)
        b = parameters.get("b", 3.0)

        return PDE(
            rhs={
                "u": f"laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
                "v": f"{Dv} * laplace(v) + {b} * u - u**2 * v",
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
        """Create initial state near homogeneous steady state."""
        if ic_type == "custom":
            a = ic_params.get("a", 2.0)
            b = ic_params.get("b", 3.0)
            noise = ic_params.get("noise", 0.1)

            # Steady state: u* = a, v* = b/a
            u_ss = a
            v_ss = b / a

            rng = np.random.default_rng(ic_params.get("seed"))
            u_data = u_ss + noise * rng.standard_normal(grid.shape)
            v_data = v_ss + noise * rng.standard_normal(grid.shape)

            u = ScalarField(grid, u_data)
            u.label = "u"
            v = ScalarField(grid, v_data)
            v.label = "v"

            return FieldCollection([u, v])

        return super().create_initial_state(grid, ic_type, ic_params, **kwargs)

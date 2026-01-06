"""Hyperbolic reaction-diffusion (Turing-wave instability)."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("turing-wave")
class TuringWavePDE(MultiFieldPDEPreset):
    """Hyperbolic reaction-diffusion with Turing-wave instability.

    Based on visualpde.com formulation:

        τ·∂²u/∂t² + ∂u/∂t = Du·∇²u + f(u,v)
        τ·∂²v/∂t² + ∂v/∂t = Dv·∇²v + g(u,v)

    where f,g are Brusselator kinetics:
        f(u,v) = a - (b+1)u + u²v
        g(u,v) = bu - u²v

    Converted to first-order system (w = ∂u/∂t, z = ∂v/∂t):
        ∂u/∂t = w
        ∂v/∂t = z
        ∂w/∂t = (1/τ)[Du·∇²u + f(u,v) - w]
        ∂z/∂t = (1/τ)[Dv·∇²v + g(u,v) - z]

    The τ term enables Turing-wave (oscillatory) instabilities for Du > Dv.

    Reference: https://visualpde.com/nonlinear-physics/turing-wave
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="turing-wave",
            category="physics",
            description="Hyperbolic Brusselator with Turing-wave instability",
            equations={
                "u": "w",
                "v": "z",
                "w": "(1/tau) * (Du*laplace(u) + a - (b+1)*u + u^2*v - w)",
                "z": "(1/tau) * (Dv*laplace(v) + b*u - u^2*v - z)",
            },
            parameters=[
                PDEParameter(
                    name="tau",
                    default=0.5,
                    description="Hyperbolic coefficient (inertia)",
                    min_value=0.1,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="Du",
                    default=2.0,
                    description="Activator diffusion (Du > Dv for wave)",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dv",
                    default=1.0,
                    description="Inhibitor diffusion",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="a",
                    default=2.0,
                    description="Brusselator parameter a",
                    min_value=0.1,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="b",
                    default=5.0,
                    description="Brusselator parameter b",
                    min_value=1.0,
                    max_value=10.0,
                ),
            ],
            num_fields=4,
            field_names=["u", "v", "w", "z"],
            reference="https://visualpde.com/nonlinear-physics/turing-wave",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        tau = parameters.get("tau", 0.5)
        Du = parameters.get("Du", 2.0)
        Dv = parameters.get("Dv", 1.0)
        a = parameters.get("a", 2.0)
        b = parameters.get("b", 5.0)

        inv_tau = 1.0 / tau

        # Hyperbolic Brusselator system
        return PDE(
            rhs={
                "u": "w",
                "v": "z",
                "w": f"{inv_tau} * ({Du}*laplace(u) + {a} - ({b}+1)*u + u**2*v - w)",
                "z": f"{inv_tau} * ({Dv}*laplace(v) + {b}*u - u**2*v - z)",
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state near Brusselator equilibrium."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.1)

        # Brusselator equilibrium: u* = a, v* = b/a
        a = ic_params.get("a", 2.0)
        b = ic_params.get("b", 5.0)

        u_eq = a
        v_eq = b / a

        u_data = u_eq + noise * np.random.randn(*grid.shape)
        v_data = v_eq + noise * np.random.randn(*grid.shape)
        w_data = np.zeros(grid.shape)  # Zero velocity initially
        z_data = np.zeros(grid.shape)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"
        w = ScalarField(grid, w_data)
        w.label = "w"
        z = ScalarField(grid, z_data)
        z.label = "z"

        return FieldCollection([u, v, w, z])

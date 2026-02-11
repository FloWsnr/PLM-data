"""Hyperbolic Brusselator for Turing wave instabilities."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("hyperbolic-brusselator")
class HyperbolicBrusselatorPDE(MultiFieldPDEPreset):
    """Hyperbolic Brusselator with Turing wave instability.

    A modified reaction-diffusion system with inertial effects that exhibits
    oscillating spatial patterns - Turing instabilities with complex eigenvalues.

    The hyperbolic Brusselator introduces a relaxation time (tau):

        tau * d2u/dt2 + du/dt = Du * laplace(u) + a - (b+1)*u + u^2*v
        tau * d2v/dt2 + dv/dt = Dv * laplace(v) + b*u - u^2*v

    Implemented as a first-order system with auxiliary variables (w, q):

        du/dt = w
        dv/dt = q
        dw/dt = (Du/tau)*laplace(u) + eps*laplace(w) + (a + u^2*v - (b+1)*u - w)/tau
        dq/dt = (Dv/tau)*laplace(v) + eps*laplace(q) + (b*u - u^2*v - q)/tau

    Key features:
        - tau > 0: Memory/inertia effects create Turing wave instabilities
        - Spatial modes oscillate in time (complex eigenvalues)
        - Can produce traveling or standing wave patterns

    References:
        Zemskov et al. (2022). arXiv:2204.13820
        Fort & Mendez (2002). Phys. Rev. Lett. 89:178101
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="hyperbolic-brusselator",
            category="biology",
            description="Hyperbolic Brusselator with Turing wave instability",
            equations={
                "u": "w",
                "v": "q",
                "w": "(Du/tau)*laplace(u) + eps*laplace(w) + (a + u^2*v - (b+1)*u - w)/tau",
                "q": "(Dv/tau)*laplace(v) + eps*laplace(q) + (b*u - u^2*v - q)/tau",
            },
            parameters=[
                PDEParameter("tau", "Relaxation time (memory parameter)"),
                PDEParameter("Du", "Activator diffusion coefficient"),
                PDEParameter("Dv", "Inhibitor diffusion coefficient"),
                PDEParameter("a", "Brusselator feed rate"),
                PDEParameter("b", "Brusselator removal rate"),
                PDEParameter("eps", "Numerical diffusion for velocity variables"),
            ],
            num_fields=4,
            field_names=["u", "v", "w", "q"],
            reference="Zemskov et al. (2022)",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        tau = parameters.get("tau", 1.0)
        Du = parameters.get("Du", 2.0)
        Dv = parameters.get("Dv", 1.0)
        a = parameters.get("a", 5.0)
        b = parameters.get("b", 9.0)
        eps = parameters.get("eps", 0.001)

        # Avoid division by zero
        if tau <= 0:
            tau = 0.001

        return PDE(
            rhs={
                "u": "w",
                "v": "q",
                "w": f"{Du/tau} * laplace(u) + {eps} * laplace(w) + ({a} + u**2 * v - ({b} + 1) * u - w) / {tau}",
                "q": f"{Dv/tau} * laplace(v) + {eps} * laplace(q) + ({b} * u - u**2 * v - q) / {tau}",
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
        """Create initial state for hyperbolic Brusselator.

        Following Visual-PDE reference:
        - u, v initialized near steady state with small perturbation
        - w, q (velocity fields) initialized to 0
        """
        if ic_type == "custom":
            a = ic_params.get("a", 5.0)
            b = ic_params.get("b", 9.0)
            noise = ic_params.get("noise", 0.1)

            # Steady state: u* = a, v* = b/a, w* = 0, q* = 0
            u_ss = a
            v_ss = b / a

            rng = np.random.default_rng(ic_params.get("seed"))
            u_data = u_ss + noise * rng.standard_normal(grid.shape)
            v_data = v_ss + noise * rng.standard_normal(grid.shape)
            # Velocity fields w, q start at 0 (consistent with Visual-PDE reference)
            w_data = np.zeros(grid.shape)
            q_data = np.zeros(grid.shape)

            u = ScalarField(grid, u_data)
            u.label = "u"
            v = ScalarField(grid, v_data)
            v.label = "v"
            w = ScalarField(grid, w_data)
            w.label = "w"
            q = ScalarField(grid, q_data)
            q.label = "q"

            return FieldCollection([u, v, w, q])

        return super().create_initial_state(grid, ic_type, ic_params, **kwargs)

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

        du/dt = laplace(u) + a + u^2/(v*(1+K*u^2)) - b*u
        dv/dt = D * laplace(v) + u^2 - c*v

    where:
        - u is the activator concentration
        - v is the inhibitor concentration
        - D > 1 is required for pattern formation
        - a is basal activator production
        - b is activator decay rate
        - c is inhibitor decay rate
        - K is the saturation constant (K=0 gives standard model)

    Key phenomena:
        - Spot patterns: default behavior (K=0)
        - Stripe/labyrinthine patterns: K~0.003
        - Stripe instability: stripes break into spots when K=0

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
                "u": "laplace(u) + a + u**2 / (v * (1 + K * u**2)) - b * u",
                "v": "D * laplace(v) + u**2 - c * v",
            },
            parameters=[
                PDEParameter("D", "Inhibitor diffusion ratio (D > 1 required)"),
                PDEParameter("a", "Basal activator production"),
                PDEParameter("b", "Activator decay rate"),
                PDEParameter("c", "Inhibitor decay rate"),
                PDEParameter("K", "Saturation constant (0=spots, ~0.003=stripes)"),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Gierer & Meinhardt (1972)",
            supported_dimensions=[1, 2, 3],
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
        K = parameters.get("K", 0.0)

        # Add small epsilon to prevent division by zero
        # When K > 0, use saturation term for stripe patterns
        if K > 0:
            u_rhs = f"laplace(u) + {a} + u**2 / ((v + 1e-10) * (1 + {K} * u**2)) - {b} * u"
        else:
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
        **kwargs,
    ) -> FieldCollection:
        """Create initial state near steady state with perturbation.

        Supports:
            - random-gaussian: Random perturbation around steady state
            - stripes: Stripe pattern in x-direction (for stripe instability demos)
        """
        noise = ic_params.get("noise", 0.01)

        # Initial values near steady state (approximate)
        u0 = ic_params.get("u0", 1.0)
        v0 = ic_params.get("v0", 1.0)

        rng = np.random.default_rng(ic_params.get("seed"))

        if ic_type == "stripes":
            # Stripe initial condition for demonstrating stripe-to-spot instability
            # u(0) = amplitude * (1 + cos(n*pi*x/L)), v(0) = v0
            n_stripes = ic_params.get("n_stripes", 4)
            amplitude = ic_params.get("amplitude", 3.0)

            x_bounds = grid.axes_bounds[0]
            Lx = x_bounds[1] - x_bounds[0]
            x_1d = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])

            ndim = len(grid.shape)
            shape = [1] * ndim
            shape[0] = grid.shape[0]
            X = x_1d.reshape(shape)

            # Randomize stripe phase if not specified
            phase = ic_params.get("phase")
            if phase is None or phase == "random":
                raise ValueError("gierer-meinhardt stripes requires phase (or random)")

            # Create stripe pattern: amplitude * (1 + cos(n*pi*x/L + phase))
            u_data = amplitude * (1 + np.cos(n_stripes * np.pi * (X - x_bounds[0]) / Lx + phase))
            u_data = np.broadcast_to(u_data, grid.shape).copy()
            # Add small perturbation to trigger instability
            u_data += noise * rng.standard_normal(grid.shape)
            v_data = v0 * np.ones(grid.shape) + noise * rng.standard_normal(grid.shape)
        elif ic_type == "custom":
            # Random perturbation around steady state
            u_data = u0 * (1 + noise * rng.standard_normal(grid.shape))
            v_data = v0 * (1 + noise * rng.standard_normal(grid.shape))
        else:
            return super().create_initial_state(grid, ic_type, ic_params, **kwargs)

        # Ensure positive values
        u_data = np.maximum(u_data, 0.01)
        v_data = np.maximum(v_data, 0.01)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

    def get_position_params(self, ic_type: str) -> set[str]:
        if ic_type in ("custom", "stripes"):
            return {"phase"}
        return super().get_position_params(ic_type)

    def resolve_ic_params(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> dict[str, Any]:
        if ic_type in ("custom", "stripes"):
            resolved = ic_params.copy()
            if ic_type == "stripes":
                if "phase" not in resolved:
                    raise ValueError("gierer-meinhardt stripes requires phase (or random)")
                if resolved["phase"] == "random":
                    rng = np.random.default_rng(resolved.get("seed"))
                    resolved["phase"] = rng.uniform(0, 2 * np.pi)
                if resolved["phase"] is None:
                    raise ValueError("gierer-meinhardt stripes requires phase (or random)")
            return resolved
        return super().resolve_ic_params(grid, ic_type, ic_params)

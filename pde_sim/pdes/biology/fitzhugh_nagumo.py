"""FitzHugh-Nagumo model for excitable and pattern-forming systems."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("fitzhugh-nagumo")
class FitzHughNagumoPDE(MultiFieldPDEPreset):
    """FitzHugh-Nagumo model for excitable media.

    A simplified model of neuronal action potentials:

        du/dt = laplace(u) + u - u^3 - v
        dv/dt = Dv * laplace(v) + e_v * (u - a_v*v - a_z)

    where:
        - u is the voltage variable (fast)
        - v is the recovery variable (slow)
        - Dv > 1 for pattern formation
        - e_v controls the timescale separation

    Key behaviors:
        - Excitability: threshold-triggered large excursions
        - Oscillations: sustained periodic behavior
        - Pattern formation: Turing-like patterns when Dv >> 1
        - Spiral waves: rotating spirals in excitable regime

    References:
        FitzHugh, R. (1961). Biophysical J., 1(6), 445-466.
        Nagumo, J. et al. (1962). Proc. IRE, 50(10), 2061-2070.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="fitzhugh-nagumo",
            category="biology",
            description="FitzHugh-Nagumo excitable media",
            equations={
                "u": "laplace(u) + u - u**3 - v",
                "v": "Dv * laplace(v) + e_v * (u - a_v * v - a_z)",
            },
            parameters=[
                PDEParameter("Dv", "Inhibitor diffusion coefficient"),
                PDEParameter("e_v", "Recovery timescale parameter"),
                PDEParameter("a_v", "Recovery slope coefficient"),
                PDEParameter("a_z", "Recovery offset parameter"),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="FitzHugh (1961), Nagumo et al. (1962)",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Dv = parameters.get("Dv", 20.0)
        e_v = parameters.get("e_v", 0.5)
        a_v = parameters.get("a_v", 1.0)
        a_z = parameters.get("a_z", -0.1)

        u_eq = "laplace(u) + u - u**3 - v"
        if Dv == 0:
            v_eq = f"{e_v} * (u - {a_v} * v - {a_z})"
        else:
            v_eq = f"{Dv} * laplace(v) + {e_v} * (u - {a_v} * v - {a_z})"

        return PDE(
            rhs={"u": u_eq, "v": v_eq},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial state - typically a localized perturbation."""
        if ic_type == "spiral-seed":
            return self._spiral_seed_init(grid, ic_params)

        # Use base class implementation for all other IC types
        return super().create_initial_state(grid, ic_type, ic_params, **kwargs)

    def _spiral_seed_init(
        self,
        grid: CartesianGrid,
        params: dict[str, Any],
    ) -> FieldCollection:
        """Initialize with a cross-field pattern that seeds spiral formation.

        Creates orthogonal half-plane steps in u and v:
        - u-step along x: left half excited, right half resting (wavefront)
        - v-step along y: bottom half recovered, top half refractory (wave break)

        This produces a wavefront that propagates only in the bottom half
        (top half is refractory), with the broken end curling into a spiral.
        """
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[0]

        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        u_rest = params["u_rest"]
        v_rest = params["v_rest"]
        u_excited = params["u_excited"]
        v_refractory = params["v_refractory"]

        x_center = x_bounds[0] + Lx / 2
        y_center = y_bounds[0] + Ly / 2

        # u-step along x: left half excited, right half resting
        u_data = np.where(X < x_center, u_excited, u_rest)
        # v-step along y: bottom half recovered, top half refractory
        v_data = np.where(Y < y_center, v_rest, v_refractory)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

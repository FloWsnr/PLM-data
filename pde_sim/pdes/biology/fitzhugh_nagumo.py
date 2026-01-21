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
                PDEParameter(
                    name="Dv",
                    default=20.0,
                    description="Inhibitor diffusion coefficient",
                    min_value=0.0,
                    max_value=100.0,
                ),
                PDEParameter(
                    name="e_v",
                    default=0.5,
                    description="Recovery timescale parameter",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="a_v",
                    default=1.0,
                    description="Recovery slope coefficient",
                    min_value=0.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="a_z",
                    default=-0.1,
                    description="Recovery offset parameter",
                    min_value=-1.0,
                    max_value=1.0,
                ),
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
        """Initialize with a broken wave that seeds spiral formation."""
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[0]

        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Create broken wavefront that will curl into spiral
        u_data = -1.0 * np.ones(grid.shape)
        v_data = -0.4 * np.ones(grid.shape)

        # Left half excitation
        mask_left = X < x_bounds[0] + Lx / 2
        # Bottom half additional condition
        mask_bottom = Y < y_bounds[0] + Ly / 2

        u_data[mask_left & mask_bottom] = 1.0
        v_data[mask_left & mask_bottom] = 0.0

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

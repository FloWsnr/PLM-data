"""FitzHugh-Nagumo excitable media model."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("fitzhugh-nagumo")
class FitzHughNagumoPDE(MultiFieldPDEPreset):
    """FitzHugh-Nagumo model for excitable media.

    Pattern-forming version from visualpde.com:

        du/dt = Du * laplace(u) + u - u³ - v
        dv/dt = Dv * laplace(v) + epsilon * (u - a_v * v - a_z)

    where:
        - u is the fast activator (membrane potential analog)
        - v is the slow inhibitor (recovery variable)
        - epsilon controls the timescale separation
        - a_v, a_z are parameters

    Produces spiral waves, target patterns, and propagating pulses.
    For Turing patterns, typically D > 1.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="fitzhugh-nagumo",
            category="biology",
            description="FitzHugh-Nagumo excitable media",
            equations={
                "u": "Du * laplace(u) + u - u**3 - v",
                "v": "Dv * laplace(v) + epsilon * (u - a_v * v - a_z)",
            },
            parameters=[
                PDEParameter(
                    name="epsilon",
                    default=0.1,
                    description="Timescale ratio (small for excitable)",
                    min_value=0.001,
                    max_value=0.5,
                ),
                PDEParameter(
                    name="a_v",
                    default=0.0,
                    description="Coefficient of v in recovery",
                    min_value=0.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="a_z",
                    default=0.0,
                    description="Constant in recovery equation",
                    min_value=-1.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="Du",
                    default=1.0,
                    description="Diffusion of u",
                    min_value=0.05,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="Dv",
                    default=10.0,
                    description="Diffusion of v (D > 1 for patterns)",
                    min_value=0.0,
                    max_value=100.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="visualpde.com FitzHugh-Nagumo pattern formation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        epsilon = parameters.get("epsilon", 0.1)
        a_v = parameters.get("a_v", 0.0)
        a_z = parameters.get("a_z", 0.0)
        Du = parameters.get("Du", 1.0)
        Dv = parameters.get("Dv", 10.0)

        u_eq = f"{Du} * laplace(u) + u - u**3 - v"
        if Dv == 0:
            v_eq = f"{epsilon} * (u - {a_v} * v - {a_z})"
        else:
            v_eq = f"{Dv} * laplace(v) + {epsilon} * (u - {a_v} * v - {a_z})"

        return PDE(
            rhs={"u": u_eq, "v": v_eq},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state - typically a localized perturbation."""
        if ic_type == "spiral-seed":
            return self._spiral_seed_init(grid, ic_params)

        # Default: rest state with perturbation
        noise = ic_params.get("noise", 0.01)

        # Rest state is approximately (u, v) = (-1, -0.4) or near nullcline intersection
        u_data = -1.0 + noise * np.random.randn(*grid.shape)
        v_data = -0.4 + noise * np.random.randn(*grid.shape)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

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

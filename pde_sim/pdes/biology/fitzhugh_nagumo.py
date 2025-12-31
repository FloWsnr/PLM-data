"""FitzHugh-Nagumo excitable media model."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("fitzhugh-nagumo")
class FitzHughNagumoPDE(MultiFieldPDEPreset):
    """FitzHugh-Nagumo model for excitable media.

    Simplified model of neuron action potentials and cardiac tissue:

        du/dt = Du * laplace(u) + u - u³/3 - v
        dv/dt = Dv * laplace(v) + epsilon * (u + a - b*v)

    where:
        - u is the fast activator (membrane potential analog)
        - v is the slow inhibitor (recovery variable)
        - epsilon controls the timescale separation
        - a, b are shape parameters

    Produces spiral waves, target patterns, and propagating pulses.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="fitzhugh-nagumo",
            category="biology",
            description="FitzHugh-Nagumo excitable media",
            equations={
                "u": "Du * laplace(u) + u - u**3/3 - v",
                "v": "Dv * laplace(v) + epsilon * (u + a - b * v)",
            },
            parameters=[
                PDEParameter(
                    name="epsilon",
                    default=0.08,
                    description="Timescale ratio (small for excitable)",
                    min_value=0.001,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="a",
                    default=0.7,
                    description="Parameter a",
                    min_value=0.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="b",
                    default=0.8,
                    description="Parameter b",
                    min_value=0.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="Du",
                    default=1.0,
                    description="Diffusion of u",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dv",
                    default=0.0,
                    description="Diffusion of v (often 0)",
                    min_value=0.0,
                    max_value=10.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Nerve impulse and cardiac tissue modeling",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        epsilon = parameters.get("epsilon", 0.08)
        a = parameters.get("a", 0.7)
        b = parameters.get("b", 0.8)
        Du = parameters.get("Du", 1.0)
        Dv = parameters.get("Dv", 0.0)

        u_eq = f"{Du} * laplace(u) + u - u**3 / 3 - v"
        v_eq = f"{Dv} * laplace(v) + {epsilon} * (u + {a} - {b} * v)"
        if Dv == 0:
            v_eq = f"{epsilon} * (u + {a} - {b} * v)"

        return PDE(
            rhs={"u": u_eq, "v": v_eq},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
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


@register_pde("allen-cahn")
class AllenCahnPDE(ScalarPDEPreset):
    """Allen-Cahn (bistable) equation.

    The Allen-Cahn equation describes phase transitions:

        du/dt = D * laplace(u) + u * (1 - u²)

    or with threshold parameter a:

        du/dt = D * laplace(u) + u * (u - a) * (1 - u)

    Exhibits bistable traveling fronts.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="allen-cahn",
            category="biology",
            description="Allen-Cahn bistable equation",
            equations={
                "u": "D * laplace(u) + u * (u - a) * (1 - u)",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Diffusion coefficient",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="a",
                    default=0.5,
                    description="Threshold parameter (0 < a < 1)",
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Bistable traveling fronts, Allee effect",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        a = parameters.get("a", 0.5)

        return PDE(
            rhs={"u": f"{D} * laplace(u) + u * (u - {a}) * (1 - u)"},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

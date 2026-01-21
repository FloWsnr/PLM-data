"""Three-species FitzHugh-Nagumo variant with competing oscillations and patterns."""

import math
from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("fitzhugh-nagumo-3")
class FitzHughNagumo3PDE(MultiFieldPDEPreset):
    """Three-species FitzHugh-Nagumo variant.

    Extends the classic FitzHugh-Nagumo with a third species, creating
    competition between oscillations and pattern formation:

        du/dt = laplace(u) + u - u^3 - v
        dv/dt = Dv*laplace(v) + e_v*(u - a_v*v - a_w*w - a_z)
        dw/dt = Dw*laplace(w) + e_w*(u - w)

    where:
        - u is the voltage/activator (fast, with cubic nonlinearity)
        - v is the first recovery variable (medium timescale)
        - w is the second recovery variable (slow, pattern-forming)
        - Dv, Dw > 1 for pattern formation

    Key behaviors:
        - Low a_v (< 0.3): Patterns formed initially are destroyed by oscillations
        - High a_v (>= 0.3): Patterns stabilize and overtake oscillations
        - Competition between local oscillations and spatial patterns

    Reference:
        https://visualpde.com/nonlinear-dynamics/fitzhugh-nagumo-3
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="fitzhugh-nagumo-3",
            category="biology",
            description="Three-species FitzHugh-Nagumo with pattern-oscillation competition",
            equations={
                "u": "laplace(u) + u - u**3 - v",
                "v": "Dv * laplace(v) + e_v * (u - a_v*v - a_w*w - a_z)",
                "w": "Dw * laplace(w) + e_w * (u - w)",
            },
            parameters=[
                PDEParameter(
                    name="Dv",
                    default=40.0,
                    description="Diffusion for v (first recovery)",
                    min_value=1.0,
                    max_value=100.0,
                ),
                PDEParameter(
                    name="Dw",
                    default=200.0,
                    description="Diffusion for w (second recovery, pattern-forming)",
                    min_value=10.0,
                    max_value=500.0,
                ),
                PDEParameter(
                    name="a_v",
                    default=0.2,
                    description="Self-inhibition of v (controls pattern vs oscillation)",
                    min_value=0.0,
                    max_value=0.5,
                ),
                PDEParameter(
                    name="e_v",
                    default=0.2,
                    description="Timescale parameter for v",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="e_w",
                    default=1.0,
                    description="Timescale parameter for w",
                    min_value=0.1,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="a_w",
                    default=0.5,
                    description="Coupling coefficient v-w",
                    min_value=0.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="a_z",
                    default=-0.1,
                    description="Offset parameter for v dynamics",
                    min_value=-1.0,
                    max_value=1.0,
                ),
            ],
            num_fields=3,
            field_names=["u", "v", "w"],
            reference="VisualPDE FitzHugh-Nagumo-3",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Dv = parameters.get("Dv", 40.0)
        Dw = parameters.get("Dw", 200.0)
        a_v = parameters.get("a_v", 0.2)
        e_v = parameters.get("e_v", 0.2)
        e_w = parameters.get("e_w", 1.0)
        a_w = parameters.get("a_w", 0.5)
        a_z = parameters.get("a_z", -0.1)

        # u equation: cubic activator dynamics
        u_eq = "laplace(u) + u - u**3 - v"

        # v equation: first recovery with coupling to w
        v_eq = f"{Dv} * laplace(v) + {e_v} * (u - {a_v}*v - {a_w}*w - {a_z})"

        # w equation: second recovery (slow, pattern-forming)
        w_eq = f"{Dw} * laplace(w) + {e_w} * (u - w)"

        return PDE(
            rhs={"u": u_eq, "v": v_eq, "w": w_eq},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial state with Gaussian u, cosine v, zero w.

        Matches Visual PDE reference:
            u = 5*exp(-0.1*((x-Lx/2)^2 + (y-Ly/2)^2))
            v = cos(m*x*pi/280)*cos(m*y*pi/280)
            w = 0
        """
        seed = ic_params.get("seed")
        if seed is not None:
            np.random.seed(seed)

        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[0]

        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        # u: Gaussian blob at center
        cx = (x_bounds[0] + x_bounds[1]) / 2
        cy = (y_bounds[0] + y_bounds[1]) / 2
        amplitude = ic_params.get("amplitude", 5.0)
        width = ic_params.get("width", 0.1)
        r2 = (X - cx) ** 2 + (Y - cy) ** 2
        u_data = amplitude * np.exp(-width * r2)

        # v: Cosine pattern
        m = ic_params.get("m", 4)
        # Use domain size as reference (Visual PDE uses 280)
        v_data = np.cos(m * X * math.pi / Lx) * np.cos(m * Y * math.pi / Ly)

        # w: Zero initially
        w_data = np.zeros(grid.shape)

        # Add small noise if requested
        noise = ic_params.get("noise", 0.0)
        if noise > 0:
            u_data += noise * np.random.randn(*grid.shape)
            v_data += noise * np.random.randn(*grid.shape)
            w_data += noise * np.random.randn(*grid.shape)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"
        w = ScalarField(grid, w_data)
        w.label = "w"

        return FieldCollection([u, v, w])

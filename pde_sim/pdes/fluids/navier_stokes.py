"""2D Navier-Stokes equations in velocity-pressure formulation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("navier-stokes")
class NavierStokesPDE(MultiFieldPDEPreset):
    """2D incompressible Navier-Stokes equations.

    The Navier-Stokes equations govern the motion of viscous fluids,
    describing how velocity and pressure fields evolve over time.

    Momentum equations:
        du/dt = -(u * d_dx(u) + v * d_dy(u)) - d_dx(p) + nu * laplace(u)
        dv/dt = -(u * d_dx(v) + v * d_dy(v)) - d_dy(p) + nu * laplace(v)

    Generalized pressure equation:
        dp/dt = nu * laplace(p) - (1/M^2) * (d_dx(u) + d_dy(v))

    Passive scalar transport:
        dS/dt = -(u * d_dx(S) + v * d_dy(S)) + D * laplace(S)

    where:
        - u, v are velocity components
        - p is pressure
        - S is passive scalar for flow visualization
        - nu is kinematic viscosity
        - M is Mach number parameter (related to pressure wave speed)
        - D is passive scalar diffusion coefficient
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="navier-stokes",
            category="fluids",
            description="2D incompressible Navier-Stokes equations",
            equations={
                "u": "-(u * d_dx(u) + v * d_dy(u)) - d_dx(p) + nu * laplace(u)",
                "v": "-(u * d_dx(v) + v * d_dy(v)) - d_dy(p) + nu * laplace(v)",
                "p": "nu * laplace(p) - (1/M^2) * (d_dx(u) + d_dy(v))",
                "S": "-(u * d_dx(S) + v * d_dy(S)) + D * laplace(S)",
            },
            parameters=[
                PDEParameter("nu", "Kinematic viscosity"),
                PDEParameter("M", "Mach number parameter"),
                PDEParameter("D", "Passive scalar diffusion coefficient"),
            ],
            num_fields=4,
            field_names=["u", "v", "p", "S"],
            reference="Navier-Stokes equations (Stokes/Navier, 1800s)",
            supported_dimensions=[2],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        nu = parameters.get("nu", 0.02)
        M = parameters.get("M", 0.5)
        D = parameters.get("D", 0.05)

        # Pressure relaxation coefficient
        inv_M2 = 1.0 / (M * M)

        return PDE(
            rhs={
                "u": f"-u * d_dx(u) - v * d_dy(u) - d_dx(p) + {nu} * laplace(u)",
                "v": f"-u * d_dx(v) - v * d_dy(v) - d_dy(p) + {nu} * laplace(v)",
                "p": f"{nu} * laplace(p) - {inv_M2} * (d_dx(u) + d_dy(v))",
                "S": f"-u * d_dx(S) - v * d_dy(S) + {D} * laplace(S)",
            },
            **self._get_pde_bc_kwargs(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial velocity, pressure, and passive scalar fields."""
        # Get domain bounds from grid
        x_min, x_max = grid.axes_bounds[0]
        y_min, y_max = grid.axes_bounds[1]
        L_x = x_max - x_min
        L_y = y_max - y_min

        if ic_type in ("navier-stokes-default", "default", "shear-layer"):
            # Shear layer initial condition
            shear_width = ic_params.get("shear_width", 0.1)
            amplitude = ic_params.get("amplitude", 0.5)

            x, y = np.meshgrid(
                np.linspace(x_min, x_max, grid.shape[0]),
                np.linspace(y_min, y_max, grid.shape[1]),
                indexing="ij",
            )

            # Normalize y to [0, 1] range
            y_norm = (y - y_min) / L_y

            # Shear layer velocity profile
            u_data = amplitude * np.tanh((y_norm - 0.5) / shear_width)
            v_data = 0.05 * np.sin(2 * np.pi * (x - x_min) / L_x)
            p_data = np.zeros_like(u_data)
            # Passive scalar gradient for visualization
            S_data = y_norm

        elif ic_type == "vortex-pair":
            # Counter-rotating vortex pair
            x0_1, y0_1 = ic_params.get("x1", 0.35), ic_params.get("y1", 0.5)
            x0_2, y0_2 = ic_params.get("x2", 0.65), ic_params.get("y2", 0.5)
            strength = ic_params.get("strength", 1.0)
            radius = ic_params.get("radius", 0.1)

            x, y = np.meshgrid(
                np.linspace(x_min, x_max, grid.shape[0]),
                np.linspace(y_min, y_max, grid.shape[1]),
                indexing="ij",
            )

            # Normalize coordinates
            x_norm = (x - x_min) / L_x
            y_norm = (y - y_min) / L_y

            # Distance from vortex centers
            r1_sq = (x_norm - x0_1) ** 2 + (y_norm - y0_1) ** 2
            r2_sq = (x_norm - x0_2) ** 2 + (y_norm - y0_2) ** 2

            # Gaussian vortices induce velocity
            # For a vortex at (x0, y0), velocity is tangential
            # u = -strength * (y - y0) / r * exp(-r^2/2*sigma^2)
            # v =  strength * (x - x0) / r * exp(-r^2/2*sigma^2)
            sigma = radius

            u_data = (
                -strength * (y_norm - y0_1) * np.exp(-r1_sq / (2 * sigma**2))
                - (-strength) * (y_norm - y0_2) * np.exp(-r2_sq / (2 * sigma**2))
            )
            v_data = (
                strength * (x_norm - x0_1) * np.exp(-r1_sq / (2 * sigma**2))
                + (-strength) * (x_norm - x0_2) * np.exp(-r2_sq / (2 * sigma**2))
            )
            p_data = np.zeros_like(u_data)
            S_data = np.exp(-r1_sq / (2 * sigma**2)) + np.exp(-r2_sq / (2 * sigma**2))

        elif ic_type == "poiseuille":
            # Poiseuille (pressure-driven channel) flow
            # Reference: Visual PDE NavierStokesPoiseuilleFlow preset
            # Parabolic velocity profile with no-slip walls at y=0 and y=L_y
            amplitude = ic_params.get("amplitude", 0.4)
            pressure_gradient = ic_params.get("pressure_gradient", 1.0)

            x, y = np.meshgrid(
                np.linspace(x_min, x_max, grid.shape[0]),
                np.linspace(y_min, y_max, grid.shape[1]),
                indexing="ij",
            )

            # Normalize coordinates
            x_norm = (x - x_min) / L_x
            y_norm = (y - y_min) / L_y

            # Parabolic velocity profile: u(y) = -amplitude * 6 * y * (1-y)
            # Peak velocity at center (y=0.5), zero at walls (y=0, y=1)
            # The factor of 6 normalizes so that the max velocity = amplitude
            u_data = -amplitude * 6.0 * y_norm * (1.0 - y_norm)

            # No vertical velocity
            v_data = np.zeros_like(u_data)

            # Linear pressure gradient driving the flow
            p_data = pressure_gradient * x_norm

            # Passive scalar as x-gradient for visualization
            S_data = x_norm

        else:
            # Default: use standard IC generator for S, zeros for velocity/pressure
            S_field = create_initial_condition(grid, ic_type, ic_params)
            S_data = S_field.data
            u_data = np.zeros_like(S_data)
            v_data = np.zeros_like(S_data)
            p_data = np.zeros_like(S_data)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"
        p = ScalarField(grid, p_data)
        p.label = "p"
        S = ScalarField(grid, S_data)
        S.label = "S"

        return FieldCollection([u, v, p, S])

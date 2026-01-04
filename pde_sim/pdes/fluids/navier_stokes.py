"""2D Navier-Stokes in vorticity-stream function formulation."""

from typing import Any

import numpy as np
from pde import CartesianGrid, ScalarField, solve_poisson_equation
from pde.pdes.base import PDEBase

from pde_sim.initial_conditions import create_initial_condition

from ..base import PDEMetadata, PDEParameter, ScalarPDEPreset
from .. import register_pde


class NavierStokesVorticityPDE(PDEBase):
    """2D Navier-Stokes in vorticity-stream function formulation.

    Solves the full vorticity transport equation:
        dw/dt = nu * laplace(w) - u * d_dx(w) - v * d_dy(w)

    where velocity is derived from stream function psi:
        laplace(psi) = -w  (solved at each timestep)
        u = d_dy(psi)
        v = -d_dx(psi)

    This is the proper 2D incompressible Navier-Stokes equations
    in vorticity-stream function form, where vorticity is advected
    by its self-induced velocity field.
    """

    def __init__(self, nu: float = 0.01, bc: list | None = None):
        """Initialize the Navier-Stokes vorticity equation.

        Args:
            nu: Kinematic viscosity.
            bc: Boundary conditions.
        """
        super().__init__()
        self.nu = nu
        self.bc = bc if bc is not None else "auto_periodic_neumann"

    def evolution_rate(self, state: ScalarField, t: float = 0) -> ScalarField:
        """Compute dw/dt using stream function for velocity.

        Args:
            state: Current vorticity field w.
            t: Current time (unused).

        Returns:
            Rate of change dw/dt.
        """
        w = state

        # Solve Poisson equation: laplace(psi) = -w for stream function
        # For periodic BC, we need zero mean vorticity. Subtracting the mean
        # doesn't affect velocities since u = grad(psi) and grad(constant) = 0.
        w_zero_mean = w - w.average
        psi = solve_poisson_equation(w_zero_mean, bc=self.bc)

        # Compute velocity from stream function
        # u = d(psi)/dy, v = -d(psi)/dx
        grad_psi = psi.gradient(bc=self.bc)
        u = grad_psi[1]  # d_dy(psi)
        v = -grad_psi[0]  # -d_dx(psi)

        # Compute vorticity gradients
        grad_w = w.gradient(bc=self.bc)
        dw_dx = grad_w[0]
        dw_dy = grad_w[1]

        # Diffusion term
        laplacian_w = w.laplace(bc=self.bc)

        # Full vorticity transport: dw/dt = nu * laplace(w) - u * dw/dx - v * dw/dy
        dw_dt = self.nu * laplacian_w - u * dw_dx - v * dw_dy

        return dw_dt


@register_pde("navier-stokes")
class NavierStokesPDE(ScalarPDEPreset):
    """2D Navier-Stokes in vorticity-stream function formulation.

    The full incompressible Navier-Stokes equations in 2D can be written
    in vorticity form:

        dw/dt = nu * laplace(w) - u * d_dx(w) - v * d_dy(w)

    where:
        - w is the vorticity (w = dv/dx - du/dy)
        - nu is the kinematic viscosity
        - (u, v) is the velocity field

    The velocity is derived from the stream function psi:
        laplace(psi) = -w  (Poisson equation, solved at each timestep)
        u = d_dy(psi)
        v = -d_dx(psi)

    This formulation automatically satisfies incompressibility (div(u) = 0)
    and captures the proper nonlinear advection of vorticity by the
    self-induced velocity field.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="navier-stokes",
            category="fluids",
            description="2D Navier-Stokes vorticity-stream function",
            equations={
                "w": "nu * laplace(w) - u * d_dx(w) - v * d_dy(w)",
                "psi": "laplace(psi) = -w (stream function)",
                "u": "d_dy(psi)",
                "v": "-d_dx(psi)",
            },
            parameters=[
                PDEParameter(
                    name="nu",
                    default=0.01,
                    description="Kinematic viscosity",
                    min_value=0.001,
                    max_value=0.1,
                ),
            ],
            num_fields=1,
            field_names=["w"],
            reference="2D incompressible Navier-Stokes",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> NavierStokesVorticityPDE:
        nu = parameters.get("nu", 0.01)
        bc_spec = self._convert_bc(bc)

        return NavierStokesVorticityPDE(nu=nu, bc=bc_spec)

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial vorticity distribution."""
        if ic_type in ("navier-stokes-default", "default", "vortex-pair"):
            # Counter-rotating vortex pair
            x0_1, y0_1 = ic_params.get("x1", 0.35), ic_params.get("y1", 0.5)
            x0_2, y0_2 = ic_params.get("x2", 0.65), ic_params.get("y2", 0.5)
            strength = ic_params.get("strength", 10.0)
            radius = ic_params.get("radius", 0.1)

            x, y = np.meshgrid(
                np.linspace(0, 1, grid.shape[0]),
                np.linspace(0, 1, grid.shape[1]),
                indexing="ij",
            )

            r1_sq = (x - x0_1) ** 2 + (y - y0_1) ** 2
            r2_sq = (x - x0_2) ** 2 + (y - y0_2) ** 2

            w1 = strength * np.exp(-r1_sq / (2 * radius**2))
            w2 = -strength * np.exp(-r2_sq / (2 * radius**2))

            data = w1 + w2
            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

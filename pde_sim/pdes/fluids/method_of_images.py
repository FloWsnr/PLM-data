"""Potential flow near a wall using method of images concept."""

from typing import Any

import numpy as np
from pde import CartesianGrid, ScalarField, solve_poisson_equation
from pde.pdes.base import PDEBase

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


class MethodOfImagesVorticityPDE(PDEBase):
    """Vortex dynamics near a wall with self-advection.

    Solves the full vorticity transport equation:
        dw/dt = nu * laplace(w) - u * d_dx(w) - v * d_dy(w)

    where velocity is derived from stream function psi:
        laplace(psi) = -w  (solved at each timestep)
        u = d_dy(psi)
        v = -d_dx(psi)

    The initial condition uses mirror vortices (method of images)
    to simulate the effect of a solid wall. With proper advection,
    vortices translate parallel to the wall due to their interaction
    with image vortices.
    """

    def __init__(self, nu: float = 0.01, bc: list | None = None):
        """Initialize the method of images vorticity equation.

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


@register_pde("method-of-images")
class MethodOfImagesPDE(ScalarPDEPreset):
    """Vortex dynamics near a wall using method of images.

    The method of images places mirror vortices to simulate solid
    wall boundary conditions in potential flow. A vortex near a wall
    has an image vortex of opposite sign on the other side of the wall.

    The equations are the 2D vorticity-stream function formulation:

        dw/dt = nu * laplace(w) - u * d_dx(w) - v * d_dy(w)

    where velocity comes from stream function:
        laplace(psi) = -w
        u = d_dy(psi), v = -d_dx(psi)

    With self-advection, vortices translate parallel to the wall
    due to their interaction with image vortices, while viscosity
    causes diffusion and weakening.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="method-of-images",
            category="fluids",
            description="Vortex near wall with self-advection",
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
                PDEParameter(
                    name="wall_distance",
                    default=0.2,
                    description="Distance of vortex from wall",
                    min_value=0.05,
                    max_value=0.4,
                ),
            ],
            num_fields=1,
            field_names=["w"],
            reference="Method of images for wall-bounded vortex flow",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> MethodOfImagesVorticityPDE:
        nu = parameters.get("nu", 0.01)
        bc_spec = self._convert_bc(bc)

        return MethodOfImagesVorticityPDE(nu=nu, bc=bc_spec)

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create vortex pair near wall (real + image)."""
        wall_dist = ic_params.get("wall_distance", 0.2)
        strength = ic_params.get("strength", 10.0)
        radius = ic_params.get("radius", 0.08)

        x, y = np.meshgrid(
            np.linspace(0, 1, grid.shape[0]),
            np.linspace(0, 1, grid.shape[1]),
            indexing="ij",
        )

        # Wall at y = 0, vortex at y = wall_dist
        # Image vortex at y = -wall_dist (reflected, opposite sign)
        y_vortex = wall_dist
        y_image = -wall_dist  # Outside domain, but affects field

        # Real vortex
        r1_sq = (x - 0.5) ** 2 + (y - y_vortex) ** 2
        w1 = strength * np.exp(-r1_sq / (2 * radius**2))

        # Image vortex (opposite circulation)
        r2_sq = (x - 0.5) ** 2 + (y - y_image) ** 2
        w2 = -strength * np.exp(-r2_sq / (2 * radius**2))

        # Add mirror on the other side for periodic BC
        r3_sq = (x - 0.5) ** 2 + (y - (1.0 - wall_dist)) ** 2
        w3 = strength * np.exp(-r3_sq / (2 * radius**2))
        r4_sq = (x - 0.5) ** 2 + (y - (1.0 + wall_dist)) ** 2
        w4 = -strength * np.exp(-r4_sq / (2 * radius**2))

        data = w1 + w2 + w3 + w4
        return ScalarField(grid, data)

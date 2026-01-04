"""Vortex dipole dynamics."""

from typing import Any

import numpy as np
from pde import CartesianGrid, ScalarField, solve_poisson_equation
from pde.pdes.base import PDEBase

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


class DipoleVorticityPDE(PDEBase):
    """Vortex dipole dynamics with self-advection.

    Solves the full vorticity transport equation:
        dw/dt = nu * laplace(w) - u * d_dx(w) - v * d_dy(w)

    where velocity is derived from stream function psi:
        laplace(psi) = -w  (solved at each timestep)
        u = d_dy(psi)
        v = -d_dx(psi)

    The dipole structure (two opposite-sign vortices) creates
    self-induced motion - the dipole translates perpendicular
    to the line connecting the two vortex centers.
    """

    def __init__(self, nu: float = 0.005, bc: list | None = None):
        """Initialize the dipole vorticity equation.

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


@register_pde("dipoles")
class DipolesPDE(ScalarPDEPreset):
    """Vortex dipole dynamics with self-propelled motion.

    A vortex dipole consists of two opposite-sign vortices that
    induce a mutual velocity field, causing the pair to translate.

    The equations are the 2D vorticity-stream function formulation:

        dw/dt = nu * laplace(w) - u * d_dx(w) - v * d_dy(w)

    where velocity comes from stream function:
        laplace(psi) = -w
        u = d_dy(psi), v = -d_dx(psi)

    The self-induced velocity causes the dipole to move perpendicular
    to the line joining the two vortex centers. With viscosity,
    the vortices also diffuse and weaken over time.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="dipoles",
            category="fluids",
            description="Self-propelled vortex dipole dynamics",
            equations={
                "w": "nu * laplace(w) - u * d_dx(w) - v * d_dy(w)",
                "psi": "laplace(psi) = -w (stream function)",
                "u": "d_dy(psi)",
                "v": "-d_dx(psi)",
            },
            parameters=[
                PDEParameter(
                    name="nu",
                    default=0.005,
                    description="Kinematic viscosity",
                    min_value=0.001,
                    max_value=0.1,
                ),
                PDEParameter(
                    name="separation",
                    default=0.15,
                    description="Vortex separation distance",
                    min_value=0.05,
                    max_value=0.3,
                ),
            ],
            num_fields=1,
            field_names=["w"],
            reference="Vortex dipole self-propulsion",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> DipoleVorticityPDE:
        nu = parameters.get("nu", 0.005)
        bc_spec = self._convert_bc(bc)

        return DipoleVorticityPDE(nu=nu, bc=bc_spec)

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create vortex dipole (two opposite-sign vortices)."""
        separation = ic_params.get("separation", 0.15)
        strength = ic_params.get("strength", 10.0)
        radius = ic_params.get("radius", 0.05)

        x, y = np.meshgrid(
            np.linspace(0, 1, grid.shape[0]),
            np.linspace(0, 1, grid.shape[1]),
            indexing="ij",
        )

        # Dipole centered at (0.3, 0.5), moving to the right
        cx, cy = 0.3, 0.5

        # Two vortices separated vertically
        y1 = cy + separation / 2
        y2 = cy - separation / 2

        r1_sq = (x - cx) ** 2 + (y - y1) ** 2
        r2_sq = (x - cx) ** 2 + (y - y2) ** 2

        # Opposite circulations create rightward motion
        w1 = strength * np.exp(-r1_sq / (2 * radius**2))
        w2 = -strength * np.exp(-r2_sq / (2 * radius**2))

        data = w1 + w2
        return ScalarField(grid, data)

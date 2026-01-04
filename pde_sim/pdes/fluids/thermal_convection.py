"""Rayleigh-Bénard thermal convection."""

from typing import Any

import numpy as np
from pde import CartesianGrid, FieldCollection, ScalarField, solve_poisson_equation
from pde.pdes.base import PDEBase

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


class ThermalConvectionStreamFunctionPDE(PDEBase):
    """Rayleigh-Bénard thermal convection with full advection.

    Coupled temperature and vorticity equations with stream function:

        dT/dt = kappa * laplace(T) - u * d_dx(T) - v * d_dy(T)
        dw/dt = nu * laplace(w) - u * d_dx(w) - v * d_dy(w) + Ra * d_dx(T)

    where velocity is derived from stream function psi:
        laplace(psi) = -w
        u = d_dy(psi)
        v = -d_dx(psi)

    The buoyancy term Ra * d_dx(T) drives convection when there are
    horizontal temperature gradients.
    """

    def __init__(
        self,
        kappa: float = 0.01,
        nu: float = 0.01,
        Ra: float = 100.0,
        bc: list | None = None,
    ):
        """Initialize thermal convection equation.

        Args:
            kappa: Thermal diffusivity.
            nu: Kinematic viscosity.
            Ra: Rayleigh number (buoyancy strength).
            bc: Boundary conditions.
        """
        super().__init__()
        self.kappa = kappa
        self.nu = nu
        self.Ra = Ra
        self.bc = bc if bc is not None else "auto_periodic_neumann"

    def evolution_rate(
        self, state: FieldCollection, t: float = 0
    ) -> FieldCollection:
        """Compute time derivatives using stream function for velocity.

        Args:
            state: FieldCollection with [T, w] fields.
            t: Current time (unused).

        Returns:
            FieldCollection with [dT/dt, dw/dt].
        """
        T = state[0]  # Temperature
        w = state[1]  # Vorticity

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

        # Temperature gradients
        grad_T = T.gradient(bc=self.bc)
        dT_dx = grad_T[0]
        dT_dy = grad_T[1]

        # Vorticity gradients
        grad_w = w.gradient(bc=self.bc)
        dw_dx = grad_w[0]
        dw_dy = grad_w[1]

        # Diffusion terms
        laplacian_T = T.laplace(bc=self.bc)
        laplacian_w = w.laplace(bc=self.bc)

        # Temperature equation with advection:
        # dT/dt = kappa * laplace(T) - u * dT/dx - v * dT/dy
        dT_dt = self.kappa * laplacian_T - u * dT_dx - v * dT_dy
        dT_dt.label = "T"

        # Vorticity equation with advection and buoyancy:
        # dw/dt = nu * laplace(w) - u * dw/dx - v * dw/dy + Ra * dT/dx
        dw_dt = self.nu * laplacian_w - u * dw_dx - v * dw_dy + self.Ra * dT_dx
        dw_dt.label = "w"

        return FieldCollection([dT_dt, dw_dt])


@register_pde("thermal-convection")
class ThermalConvectionPDE(MultiFieldPDEPreset):
    """Rayleigh-Bénard thermal convection with full advection.

    The Boussinesq approximation equations for thermal convection:

        dT/dt = kappa * laplace(T) - u * d_dx(T) - v * d_dy(T)
        dw/dt = nu * laplace(w) - u * d_dx(w) - v * d_dy(w) + Ra * d_dx(T)

    where:
        - T is temperature
        - w is vorticity
        - Ra is Rayleigh number (buoyancy driving)
        - kappa is thermal diffusivity
        - nu is kinematic viscosity
        - (u, v) is velocity from stream function

    The velocity is derived from stream function psi:
        laplace(psi) = -w (solved at each timestep)
        u = d_dy(psi)
        v = -d_dx(psi)

    The buoyancy term Ra * d_dx(T) couples temperature to vorticity,
    driving convection when horizontal temperature gradients exist.
    This is the mechanism behind Rayleigh-Bénard convection rolls.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="thermal-convection",
            category="fluids",
            description="Rayleigh-Bénard convection with advection",
            equations={
                "T": "kappa * laplace(T) - u * d_dx(T) - v * d_dy(T)",
                "w": "nu * laplace(w) - u * d_dx(w) - v * d_dy(w) + Ra * d_dx(T)",
                "psi": "laplace(psi) = -w (stream function)",
                "u": "d_dy(psi)",
                "v": "-d_dx(psi)",
            },
            parameters=[
                PDEParameter(
                    name="kappa",
                    default=0.01,
                    description="Thermal diffusivity",
                    min_value=0.001,
                    max_value=0.1,
                ),
                PDEParameter(
                    name="nu",
                    default=0.01,
                    description="Kinematic viscosity",
                    min_value=0.001,
                    max_value=0.1,
                ),
                PDEParameter(
                    name="Ra",
                    default=100.0,
                    description="Rayleigh number",
                    min_value=10.0,
                    max_value=1000.0,
                ),
            ],
            num_fields=2,
            field_names=["T", "w"],
            reference="Rayleigh-Bénard convection rolls",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> ThermalConvectionStreamFunctionPDE:
        kappa = parameters.get("kappa", 0.01)
        nu = parameters.get("nu", 0.01)
        Ra = parameters.get("Ra", 100.0)
        bc_spec = self._convert_bc(bc)

        return ThermalConvectionStreamFunctionPDE(
            kappa=kappa, nu=nu, Ra=Ra, bc=bc_spec
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial temperature gradient with perturbation."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.1)

        x, y = np.meshgrid(
            np.linspace(0, 1, grid.shape[0]),
            np.linspace(0, 1, grid.shape[1]),
            indexing="ij",
        )

        # Linear temperature profile (hot bottom, cold top) with perturbation
        T_data = 1.0 - y + noise * np.random.randn(*grid.shape)

        # Small initial vorticity
        w_data = noise * np.random.randn(*grid.shape)

        T = ScalarField(grid, T_data)
        T.label = "T"
        w = ScalarField(grid, w_data)
        w.label = "w"

        return FieldCollection([T, w])

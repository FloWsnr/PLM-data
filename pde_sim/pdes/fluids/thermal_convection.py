"""Rayleigh-Benard thermal convection."""

from typing import Any

import numpy as np
from pde import PDEBase, CartesianGrid, FieldCollection, ScalarField

from pde_sim.boundaries import BoundaryConditionFactory
from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


class ThermalConvectionPDEImpl(PDEBase):
    """Custom PDEBase for thermal convection with per-field boundary conditions.

    This implementation allows different BCs for different fields:
    - omega/psi: dirichlet (no-slip walls)
    - b: neumann at bottom (heat flux), dirichlet at top (cold boundary)
    """

    def __init__(self, nu: float, epsilon: float, kappa: float,
                 bc_omega: Any, bc_psi: Any, bc_b: Any):
        """Initialize thermal convection PDE.

        Args:
            nu: Kinematic viscosity
            epsilon: Relaxation parameter for stream function
            kappa: Thermal diffusivity
            bc_omega: Boundary condition for vorticity field
            bc_psi: Boundary condition for stream function field
            bc_b: Boundary condition for temperature field
        """
        super().__init__()
        self.nu = nu
        self.epsilon = epsilon
        self.kappa = kappa
        self.inv_eps = 1.0 / epsilon
        self.bc_omega = bc_omega
        self.bc_psi = bc_psi
        self.bc_b = bc_b

    def evolution_rate(self, state, t=0):
        """Evaluate the right hand side of the PDE (numpy backend).

        Equations:
            d(omega)/dt = nu * laplace(omega) - d_dy(psi) * d_dx(omega)
                          + d_dx(psi) * d_dy(omega) + d_dx(b)
            d(psi)/dt = (laplace(psi) + omega) / epsilon
            d(b)/dt = kappa * laplace(b) - d_dy(psi) * d_dx(b)
                      + d_dx(psi) * d_dy(b)
        """
        omega, psi, b = state
        rhs = state.copy()

        # Compute gradients (each field uses its own BC)
        grad_psi = psi.gradient(self.bc_psi)
        psi_x = grad_psi[0]  # d_dx(psi)
        psi_y = grad_psi[1]  # d_dy(psi)

        grad_omega = omega.gradient(self.bc_omega)
        omega_x = grad_omega[0]
        omega_y = grad_omega[1]

        grad_b = b.gradient(self.bc_b)
        b_x = grad_b[0]
        b_y = grad_b[1]

        # omega: vorticity evolution with buoyancy forcing
        rhs[0] = (self.nu * omega.laplace(self.bc_omega)
                  - psi_y * omega_x + psi_x * omega_y
                  + b_x)

        # psi: stream function relaxation
        rhs[1] = self.inv_eps * (psi.laplace(self.bc_psi) + omega)

        # b: temperature evolution with advection
        rhs[2] = (self.kappa * b.laplace(self.bc_b)
                  - psi_y * b_x + psi_x * b_y)

        return rhs

    # Note: make_pde_rhs_numba is not implemented because numba has issues
    # with multi-field FieldCollection indexing. The numpy backend (via
    # evolution_rate) works correctly and is used automatically.


@register_pde("thermal-convection")
class ThermalConvectionPDE(MultiFieldPDEPreset):
    """Rayleigh-Benard thermal convection (Boussinesq model).

    This system models buoyancy-driven fluid motion from temperature gradients.
    When a fluid layer is heated from below, it becomes unstable and develops
    organized convection cells (Benard cells).

    Vorticity evolution with buoyancy forcing:
        d(omega)/dt = nu * laplace(omega) - d_dy(psi) * d_dx(omega) + d_dx(psi) * d_dy(omega) + d_dx(b)

    Stream function (relaxation form):
        epsilon * d(psi)/dt = laplace(psi) + omega

    Temperature perturbation:
        db/dt = kappa * laplace(b) - (d_dy(psi) * d_dx(b) - d_dx(psi) * d_dy(b))

    Velocity from stream function:
        u = d_dy(psi), v = -d_dx(psi)

    where:
        - omega is vorticity
        - psi is stream function
        - b is temperature perturbation from top boundary
        - nu is kinematic viscosity
        - epsilon is relaxation parameter
        - kappa is thermal diffusivity
        - T_b is bottom boundary temperature flux
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="thermal-convection",
            category="fluids",
            description="Rayleigh-Benard thermal convection",
            equations={
                "omega": "nu * laplace(omega) - d_dy(psi) * d_dx(omega) + d_dx(psi) * d_dy(omega) + d_dx(b)",
                "psi": "(laplace(psi) + omega) / epsilon",
                "b": "kappa * laplace(b) - d_dy(psi) * d_dx(b) + d_dx(psi) * d_dy(b)",
            },
            parameters=[
                PDEParameter("nu", "Kinematic viscosity"),
                PDEParameter("epsilon", "Relaxation parameter for stream function"),
                PDEParameter("kappa", "Thermal diffusivity"),
                PDEParameter("T_b", "Bottom boundary temperature flux"),
            ],
            num_fields=3,
            field_names=["omega", "psi", "b"],
            reference="Rayleigh-Benard convection (Boussinesq, Oberbeck)",
            supported_dimensions=[2],
        )

    def _get_field_bc(self, bc: Any, field_name: str) -> Any:
        """Extract boundary condition for a specific field.

        Args:
            bc: Boundary configuration (BoundaryConfig or dict)
            field_name: Name of the field ("omega", "psi", or "b")

        Returns:
            py-pde compatible BC specification
        """
        from pde_sim.core.config import BoundaryConfig

        if isinstance(bc, BoundaryConfig):
            # Get field BC dict with x-, x+, y-, y+ keys (already merged with defaults)
            field_bc = bc.get_field_bc(field_name)
            return BoundaryConditionFactory.convert_field_bc(field_bc)
        elif isinstance(bc, dict):
            # Handle dict format (from YAML config)
            # Build base BC dict from x-, x+, y-, y+ keys
            base_bc = {
                "x-": bc.get("x-", "periodic"),
                "x+": bc.get("x+", "periodic"),
                "y-": bc.get("y-", "periodic"),
                "y+": bc.get("y+", "periodic"),
            }
            # Apply per-field overrides if present
            if "fields" in bc and field_name in bc["fields"]:
                base_bc.update(bc["fields"][field_name])
            return BoundaryConditionFactory.convert_field_bc(base_bc)
        else:
            # Fallback: assume it's already in py-pde format
            return bc

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: Any,
        grid: CartesianGrid,
    ) -> ThermalConvectionPDEImpl:
        """Create the thermal convection PDE with per-field BCs.

        Args:
            parameters: Dictionary of parameter values.
            bc: Boundary condition specification (BoundaryConfig).
            grid: The computational grid.

        Returns:
            Configured ThermalConvectionPDEImpl instance.
        """
        nu = parameters.get("nu", 0.2)
        epsilon = parameters.get("epsilon", 0.05)
        kappa = parameters.get("kappa", 0.5)

        # Extract per-field BCs
        bc_omega = self._get_field_bc(bc, "omega")
        bc_psi = self._get_field_bc(bc, "psi")
        bc_b = self._get_field_bc(bc, "b")

        return ThermalConvectionPDEImpl(
            nu=nu,
            epsilon=epsilon,
            kappa=kappa,
            bc_omega=bc_omega,
            bc_psi=bc_psi,
            bc_b=bc_b,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial temperature perturbation with small noise."""
        # Get domain bounds from grid
        x_min, x_max = grid.axes_bounds[0]
        y_min, y_max = grid.axes_bounds[1]
        L_y = y_max - y_min

        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.000001)  # Very small noise per Visual PDE

        x, y = np.meshgrid(
            np.linspace(x_min, x_max, grid.shape[0]),
            np.linspace(y_min, y_max, grid.shape[1]),
            indexing="ij",
        )

        # Normalize y coordinate
        y_norm = (y - y_min) / L_y

        if ic_type in ("thermal-convection-default", "default", "linear-gradient"):
            # Small random perturbations (instability seeds)
            omega_data = noise * np.random.randn(*grid.shape)
            psi_data = np.zeros_like(omega_data)
            # Small temperature perturbations near bottom (where heating occurs)
            b_data = noise * np.random.randn(*grid.shape) * (1.0 - y_norm)

        elif ic_type == "warm-blob":
            # Localized warm region
            x0 = ic_params.get("x0", 0.5)
            y0 = ic_params.get("y0", 0.2)
            amplitude = ic_params.get("amplitude", 0.5)
            radius = ic_params.get("radius", 0.1)

            # Normalize x coordinate
            x_norm = (x - x_min) / (x_max - x_min)

            r_sq = (x_norm - x0) ** 2 + (y_norm - y0) ** 2
            b_data = amplitude * np.exp(-r_sq / (2 * radius**2))
            omega_data = noise * np.random.randn(*grid.shape)
            psi_data = np.zeros_like(omega_data)

        else:
            # Default: use standard IC generator for b, small noise for omega
            b_field = create_initial_condition(grid, ic_type, ic_params)
            b_data = b_field.data
            omega_data = noise * np.random.randn(*grid.shape)
            psi_data = np.zeros_like(omega_data)

        omega = ScalarField(grid, omega_data)
        omega.label = "omega"
        psi = ScalarField(grid, psi_data)
        psi.label = "psi"
        b = ScalarField(grid, b_data)
        b.label = "b"

        return FieldCollection([omega, psi, b])

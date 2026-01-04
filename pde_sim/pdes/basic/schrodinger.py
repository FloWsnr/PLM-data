"""Schrodinger equation and plate equation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter, ScalarPDEPreset
from .. import register_pde


@register_pde("schrodinger")
class SchrodingerPDE(ScalarPDEPreset):
    """Free-particle Schrodinger equation.

    The Schrodinger equation describes quantum mechanical wave functions:

        i * hbar * dpsi/dt = -hbar^2/(2m) * laplace(psi)

    In dimensionless form with hbar=m=1:

        dpsi/dt = i * D * laplace(psi)

    where:
        - psi is the complex wave function
        - D is the diffusion-like coefficient (hbar/2m in physical units)

    The wave function is complex-valued.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="schrodinger",
            category="basic",
            description="Free-particle Schrodinger equation",
            equations={"psi": "1j * D * laplace(psi)"},
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.3,
                    description="Diffusion coefficient (hbar/2m)",
                    min_value=0.1,
                    max_value=0.5,
                ),
            ],
            num_fields=1,
            field_names=["psi"],
            reference="Quantum mechanics wave equation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Schrodinger equation PDE.

        Args:
            parameters: Dictionary containing 'D' coefficient.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        D = parameters.get("D", 1.0)

        # Convert BC to py-pde format
        bc_spec = self._convert_bc(bc)

        return PDE(
            rhs={"psi": f"1j * {D} * laplace(psi)"},
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial complex wave function.

        Default is a Gaussian wave packet.
        """
        if ic_type == "schrodinger-default" or ic_type == "wave-packet":
            # Gaussian wave packet
            x0 = ic_params.get("x0", 0.5)
            y0 = ic_params.get("y0", 0.5)
            sigma = ic_params.get("sigma", 0.1)
            kx = ic_params.get("kx", 3.0)  # wave vector
            ky = ic_params.get("ky", 0.0)

            # Get coordinates
            x, y = np.meshgrid(
                np.linspace(0, 1, grid.shape[0]),
                np.linspace(0, 1, grid.shape[1]),
                indexing="ij",
            )

            # Gaussian envelope with plane wave
            envelope = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
            phase = np.exp(1j * (kx * x + ky * y))
            psi_data = envelope * phase

            # Normalize
            norm = np.sqrt(np.sum(np.abs(psi_data) ** 2))
            if norm > 0:
                psi_data = psi_data / norm

            return ScalarField(grid, psi_data, dtype=complex)

        # Fall back to parent implementation for standard IC types
        field = create_initial_condition(grid, ic_type, ic_params)
        # Convert to complex
        return ScalarField(grid, field.data.astype(complex), dtype=complex)


@register_pde("plate")
class PlatePDE(MultiFieldPDEPreset):
    """Plate vibration equation (wave form).

    The plate equation describes thin elastic plate vibrations using the
    biharmonic operator:

        d²u/dt² = -D * laplace(laplace(u)) - C * du/dt

    Converted to first-order system:
        du/dt = v
        dv/dt = -D * laplace(laplace(u)) - C * v

    where:
        - u is plate displacement
        - v is velocity
        - D is the bending stiffness
        - C is damping coefficient

    With uniform initial displacement and Dirichlet boundaries, this creates
    compression waves propagating inward from the edges.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="plate",
            category="basic",
            description="Plate vibration equation (biharmonic wave)",
            equations={
                "u": "v",
                "v": "-D * laplace(laplace(u)) - C * v",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.0001,
                    description="Bending stiffness coefficient",
                    min_value=0.00001,
                    max_value=0.001,
                ),
                PDEParameter(
                    name="C",
                    default=0.5,
                    description="Damping coefficient",
                    min_value=0.0,
                    max_value=2.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Kirchhoff-Love plate theory",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the plate vibration PDE.

        Args:
            parameters: Dictionary containing 'D' and 'C' coefficients.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        D = parameters.get("D", 0.0001)
        C = parameters.get("C", 0.5)

        bc_spec = self._convert_bc(bc)

        # Build v equation with optional damping
        v_rhs = f"-{D} * laplace(laplace(u))"
        if C > 0:
            v_rhs += f" - {C} * v"

        return PDE(
            rhs={
                "u": "v",
                "v": v_rhs,
            },
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state for plate equation.

        Args:
            grid: The computational grid.
            ic_type: Type of initial condition.
            ic_params: Parameters for the initial condition.

        Returns:
            FieldCollection with u (displacement) and v (velocity) fields.
        """
        # u gets the specified initial condition
        u = create_initial_condition(grid, ic_type, ic_params)
        u.label = "u"

        # v (velocity) starts at zero by default
        v_data = np.zeros(grid.shape)
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

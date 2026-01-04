"""Schrodinger equation for quantum mechanics."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import PDEMetadata, PDEParameter, ScalarPDEPreset
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
class PlatePDE(ScalarPDEPreset):
    """Biharmonic plate equation.

    The plate equation (or biharmonic equation) describes thin plate vibrations
    and fourth-order diffusion:

        du/dt = -D * laplace(laplace(u))

    or in the wave form:

        d^2u/dt^2 = -D * laplace(laplace(u))

    This implementation uses the first-order (diffusive) form for simplicity.
    The bilaplacian creates smoothing patterns with fourth-order spatial derivatives.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="plate",
            category="basic",
            description="Biharmonic plate equation (fourth-order diffusion)",
            equations={"u": "-D * laplace(laplace(u))"},
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.1,
                    description="Biharmonic diffusion coefficient",
                    min_value=0.001,
                    max_value=0.5,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Fourth-order parabolic PDE",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the plate equation PDE.

        Args:
            parameters: Dictionary containing 'D' coefficient.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        D = parameters.get("D", 0.01)

        bc_spec = self._convert_bc(bc)

        return PDE(
            rhs={"u": f"-{D} * laplace(laplace(u))"},
            bc=bc_spec,
        )

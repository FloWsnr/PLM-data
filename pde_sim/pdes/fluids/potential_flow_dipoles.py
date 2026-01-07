"""Potential flow dipoles - source-sink pair dynamics."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("potential-flow-dipoles")
class PotentialFlowDipolesPDE(MultiFieldPDEPreset):
    """Dipole (doublet) flows in 2D potential flow.

    In potential flow theory, singularity solutions represent idealized point
    forces, vortices, sources, and sinks. A dipole arises when two equal and
    opposite singularities are brought infinitesimally close together while
    maintaining a finite product of strength and separation.

    The simulation uses parabolic relaxation to solve the elliptic Laplace
    equation for potential flow.

    Potential equation (Laplace with source-sink forcing):
        d(phi)/dt = laplace(phi) + source_forcing

    where source_forcing represents the source-sink pair.

    Velocity from potential:
        u = d_dx(phi), v = d_dy(phi)

    where:
        - phi is velocity potential
        - s is indicator field for source/sink locations
        - d is separation parameter between source and sink
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="potential-flow-dipoles",
            category="fluids",
            description="Potential flow dipole (source-sink pair)",
            equations={
                "phi": "laplace(phi) + source_forcing",
                "s": "0 (indicator field)",
            },
            parameters=[
                PDEParameter(
                    name="d",
                    default=5.0,
                    description="Separation parameter",
                    min_value=1.0,
                    max_value=30.0,
                ),
                PDEParameter(
                    name="strength",
                    default=1000.0,
                    description="Source/sink strength",
                    min_value=100.0,
                    max_value=5000.0,
                ),
            ],
            num_fields=2,
            field_names=["phi", "s"],
            reference="Potential flow theory (Laplace equation)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        # For potential flow dipoles, we solve Laplace with source-sink forcing
        # The s field indicates source (+) and sink (-) locations
        strength = parameters.get("strength", 1000.0)
        return PDE(
            rhs={
                "phi": f"laplace(phi) + {strength} * s",
                "s": "0",  # Indicator field doesn't evolve
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial potential with source-sink pair."""
        # Get domain bounds from grid
        x_min, x_max = grid.axes_bounds[0]
        y_min, y_max = grid.axes_bounds[1]
        L_x = x_max - x_min
        L_y = y_max - y_min

        d = ic_params.get("d", 5.0)
        strength = ic_params.get("strength", 1000.0)

        x, y = np.meshgrid(
            np.linspace(x_min, x_max, grid.shape[0]),
            np.linspace(y_min, y_max, grid.shape[1]),
            indexing="ij",
        )

        # Center of domain
        cx = x_min + 0.5 * L_x
        cy = y_min + 0.5 * L_y

        if ic_type in ("potential-flow-dipoles-default", "default", "dipole"):
            # Source-sink pair separated by distance 2*d*dx in x-direction
            dx = L_x / grid.shape[0]
            separation = d * dx

            # Source at (cx + separation, cy), sink at (cx - separation, cy)
            r_source_sq = (x - (cx + separation)) ** 2 + (y - cy) ** 2
            r_sink_sq = (x - (cx - separation)) ** 2 + (y - cy) ** 2

            # Avoid division by zero at singularities
            eps = (dx * 0.1) ** 2
            r_source_sq = np.maximum(r_source_sq, eps)
            r_sink_sq = np.maximum(r_sink_sq, eps)

            # Potential from source-sink pair (log singularities)
            phi_data = (strength / (4 * np.pi)) * (
                np.log(r_sink_sq) - np.log(r_source_sq)
            )

            # Indicator field showing source/sink locations
            radius = dx * 2
            s_data = np.exp(-r_source_sq / (2 * radius**2)) - np.exp(
                -r_sink_sq / (2 * radius**2)
            )

        else:
            # Default: zero potential with no sources
            phi_data = np.zeros(grid.shape)
            s_data = np.zeros(grid.shape)

        phi = ScalarField(grid, phi_data)
        phi.label = "phi"
        s = ScalarField(grid, s_data)
        s.label = "s"

        return FieldCollection([phi, s])

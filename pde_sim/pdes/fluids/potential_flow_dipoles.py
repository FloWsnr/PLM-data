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

    Visual PDE reference equation:
        d(phi)/dt = laplace(phi) + 1000*(ind(s[x-d*dx,y]>0) - ind(s[x+d*dx,y]>0))/(2*d*dx)

    Since py-pde doesn't support shifted field access, we pre-compute the
    dipole forcing term in the s field initial condition:
        s = (bump(x-d*dx,y) - bump(x+d*dx,y)) / (2*d*dx)

    Then the equation simplifies to:
        d(phi)/dt = laplace(phi) + strength * s

    Velocity from potential:
        u = d_dx(phi), v = d_dy(phi)

    where:
        - phi is velocity potential (starts at 0, relaxes to steady state)
        - s is pre-computed dipole forcing field
        - d is separation parameter between source and sink
        - strength controls the dipole intensity (default 1000)
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
        """Create initial potential with source-sink pair.

        Following Visual PDE approach:
        - phi starts at 0 (relaxes to steady state via parabolic relaxation)
        - s is a single centered bump used as source location marker
        - The dipole forcing is computed from s via shifted indicators

        Note: Our equation approximates the dipole by pre-computing the shifted
        indicator difference in s, since py-pde doesn't support field shifting.
        """
        # Get domain bounds from grid
        x_min, x_max = grid.axes_bounds[0]
        y_min, y_max = grid.axes_bounds[1]
        L_x = x_max - x_min
        L_y = y_max - y_min

        d = ic_params.get("d", 5.0)

        x, y = np.meshgrid(
            np.linspace(x_min, x_max, grid.shape[0]),
            np.linspace(y_min, y_max, grid.shape[1]),
            indexing="ij",
        )

        # Center of domain
        cx = x_min + 0.5 * L_x
        cy = y_min + 0.5 * L_y
        dx = L_x / grid.shape[0]

        if ic_type in ("potential-flow-dipoles-default", "default", "dipole"):
            # Visual PDE approach: phi starts at 0, relaxes to steady state
            phi_data = np.zeros(grid.shape)

            # Bump function centered at domain center with radius L/100
            L = min(L_x, L_y)
            radius = L / 100.0
            r_sq = (x - cx) ** 2 + (y - cy) ** 2

            # Create dipole forcing by computing shifted indicator difference
            # Visual PDE: (ind(s[x-d*dx,y]>0) - ind(s[x+d*dx,y]>0))/(2*d*dx)
            # The s[x-d*dx] lookup is nonzero when x is near (cx+d*dx), so:
            # - First term is positive near (cx+d*dx) = RIGHT of center (source)
            # - Second term is positive near (cx-d*dx) = LEFT of center (sink)
            # Net: source at right, sink at left
            separation = d * dx
            r_right_sq = (x - (cx + separation)) ** 2 + (y - cy) ** 2
            r_left_sq = (x - (cx - separation)) ** 2 + (y - cy) ** 2

            # Source (+) at right position, sink (-) at left position
            bump_right = np.where(r_right_sq < radius**2, 1.0, 0.0)
            bump_left = np.where(r_left_sq < radius**2, 1.0, 0.0)
            s_data = (bump_right - bump_left) / (2 * d * dx)

        else:
            # Default: zero potential with no sources
            phi_data = np.zeros(grid.shape)
            s_data = np.zeros(grid.shape)

        phi = ScalarField(grid, phi_data)
        phi.label = "phi"
        s = ScalarField(grid, s_data)
        s.label = "s"

        return FieldCollection([phi, s])

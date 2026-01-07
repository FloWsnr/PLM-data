"""Potential flow - Method of Images."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("potential-flow-images")
class PotentialFlowImagesPDE(MultiFieldPDEPreset):
    """Potential flow using the method of images technique.

    The method of images is an elegant analytical technique for enforcing
    boundary conditions in potential flow. When a singularity exists in
    an unbounded domain, placing an appropriate "image" singularity outside
    the physical domain can exactly satisfy certain boundary conditions.

    For potential flow in a half-space with a no-flux condition at the wall,
    a source requires an image source to cancel the normal velocity at the wall.

    Laplace equation (potential flow):
        d(phi)/dt = laplace(phi) + source_forcing

    No-flux boundary condition at wall:
        u|_wall = d_dx(phi)|_wall = 0

    Velocity from potential:
        u = d_dx(phi), v = d_dy(phi)

    where:
        - phi is velocity potential
        - s is indicator field for source locations
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="potential-flow-images",
            category="fluids",
            description="Potential flow with method of images",
            equations={
                "phi": "laplace(phi) + source_forcing",
                "s": "0 (indicator field)",
            },
            parameters=[
                PDEParameter(
                    name="strength",
                    default=10.0,
                    description="Source strength",
                    min_value=1.0,
                    max_value=100.0,
                ),
                PDEParameter(
                    name="wall_x",
                    default=0.5,
                    description="Wall position (x-coordinate fraction)",
                    min_value=0.3,
                    max_value=0.7,
                ),
            ],
            num_fields=2,
            field_names=["phi", "s"],
            reference="Method of images for wall-bounded potential flow",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        # Laplace relaxation for potential flow
        return PDE(
            rhs={
                "phi": "laplace(phi)",
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
        """Create initial potential with source and its image."""
        # Get domain bounds from grid
        x_min, x_max = grid.axes_bounds[0]
        y_min, y_max = grid.axes_bounds[1]
        L_x = x_max - x_min
        L_y = y_max - y_min

        strength = ic_params.get("strength", 10.0)
        wall_x_frac = ic_params.get("wall_x", 0.5)

        x, y = np.meshgrid(
            np.linspace(x_min, x_max, grid.shape[0]),
            np.linspace(y_min, y_max, grid.shape[1]),
            indexing="ij",
        )

        # Wall position
        wall_x = x_min + wall_x_frac * L_x

        if ic_type in ("potential-flow-images-default", "default", "source-with-image"):
            # Source at (3/4 * L_x, L_y/2) in the right half
            source_x = x_min + 0.75 * L_x
            source_y = y_min + 0.5 * L_y

            # Image source at reflected position (to create no-flux at wall)
            # Image is at same distance on other side of wall
            image_x = 2 * wall_x - source_x
            image_y = source_y

            dx = L_x / grid.shape[0]
            eps = (dx * 0.1) ** 2

            # Distance from source and image
            r_source_sq = (x - source_x) ** 2 + (y - source_y) ** 2
            r_image_sq = (x - image_x) ** 2 + (y - image_y) ** 2

            # Avoid division by zero at singularities
            r_source_sq = np.maximum(r_source_sq, eps)
            r_image_sq = np.maximum(r_image_sq, eps)

            # Potential from source and its image (both sources, same sign)
            # This creates no normal flow at the wall
            phi_data = -(strength / (4 * np.pi)) * (
                np.log(r_source_sq) + np.log(r_image_sq)
            )

            # Indicator field showing source locations
            radius = dx * 3
            s_data = np.exp(-r_source_sq / (2 * radius**2)) + np.exp(
                -r_image_sq / (2 * radius**2)
            )

        elif ic_type == "source-only":
            # Just the source without image (for comparison)
            source_x = x_min + 0.75 * L_x
            source_y = y_min + 0.5 * L_y

            dx = L_x / grid.shape[0]
            eps = (dx * 0.1) ** 2

            r_source_sq = (x - source_x) ** 2 + (y - source_y) ** 2
            r_source_sq = np.maximum(r_source_sq, eps)

            phi_data = -(strength / (4 * np.pi)) * np.log(r_source_sq)

            radius = dx * 3
            s_data = np.exp(-r_source_sq / (2 * radius**2))

        else:
            # Default: zero potential
            phi_data = np.zeros(grid.shape)
            s_data = np.zeros(grid.shape)

        phi = ScalarField(grid, phi_data)
        phi.label = "phi"
        s = ScalarField(grid, s_data)
        s.label = "s"

        return FieldCollection([phi, s])

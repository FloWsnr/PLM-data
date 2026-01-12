"""Potential flow - Method of Images with optional moving sources."""

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

    This implementation supports both static and moving sources:
    - Static: Source and image fixed, system relaxes to steady state
    - Parallel: Source oscillates parallel to wall, image mirrors the motion
    - Perpendicular: Source oscillates toward/away from wall, image mirrors
    - Circular: Source orbits in its half-space, image orbits in opposite half

    Laplace equation (potential flow):
        d(phi)/dt = laplace(phi) + source_forcing

    No-flux boundary condition at wall:
        u|_wall = d_dx(phi)|_wall = 0

    Velocity from potential:
        u = d_dx(phi), v = d_dy(phi)

    where:
        - phi is velocity potential
        - For static: s is indicator field for source locations
        - For moving: Gaussian sources computed in PDE equations
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="potential-flow-images",
            category="fluids",
            description="Potential flow with method of images",
            equations={
                "phi": "laplace(phi) + source_forcing",
                "s": "0 (indicator field, static only)",
            },
            parameters=[
                PDEParameter(
                    name="strength",
                    default=10.0,
                    description="Source strength",
                    min_value=1.0,
                    max_value=500.0,
                ),
                PDEParameter(
                    name="wall_x",
                    default=0.5,
                    description="Wall position (x-coordinate fraction)",
                    min_value=0.3,
                    max_value=0.7,
                ),
                PDEParameter(
                    name="sigma",
                    default=1.0,
                    description="Gaussian width for moving sources",
                    min_value=0.5,
                    max_value=3.0,
                ),
                PDEParameter(
                    name="omega",
                    default=1.0,
                    description="Angular velocity for motion (rad/time)",
                    min_value=0.1,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="amplitude",
                    default=5.0,
                    description="Amplitude of oscillation or orbit radius",
                    min_value=1.0,
                    max_value=15.0,
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
        init_params: dict[str, Any] | None = None,
        **kwargs,
    ) -> PDE:
        """Create PDE with optional moving sources."""
        strength = parameters.get("strength", 10.0)
        wall_x_frac = parameters.get("wall_x", 0.5)
        sigma = parameters.get("sigma", 1.0)
        omega = parameters.get("omega", 1.0)
        amplitude = parameters.get("amplitude", 5.0)

        # Motion type comes from init_params
        init_params = init_params or {}
        motion = init_params.get("motion", "static")

        # Get domain info
        x_min, x_max = grid.axes_bounds[0]
        y_min, y_max = grid.axes_bounds[1]
        L_x = x_max - x_min
        L_y = y_max - y_min
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        # Wall position
        wall_x = x_min + wall_x_frac * L_x

        # Base source position (in right half relative to wall)
        base_source_x = x_min + 0.75 * L_x
        base_source_y = cy

        if motion == "static":
            # Static case: use the s field for source locations
            return PDE(
                rhs={
                    "phi": f"laplace(phi) - {strength} * s",
                    "s": "0",  # Indicator field doesn't evolve
                },
                **self._get_pde_bc_kwargs(bc),
            )

        # For moving sources, compute Gaussian forcing directly in PDE
        norm = 1.0 / (2 * np.pi * sigma**2)

        if motion == "parallel":
            # Source oscillates parallel to wall (y direction)
            # Image mirrors at same y position
            source_x = base_source_x
            source_y = f"({cy} + {amplitude} * sin({omega} * t))"
            image_x = 2 * wall_x - base_source_x
            image_y = source_y  # Same y motion

            gaussian_source = f"exp(-((x - {source_x})**2 + (y - {source_y})**2) / (2 * {sigma}**2))"
            gaussian_image = f"exp(-((x - {image_x})**2 + (y - {image_y})**2) / (2 * {sigma}**2))"

        elif motion == "perpendicular":
            # Source oscillates perpendicular to wall (x direction)
            # Must stay in right half, so oscillate around base position
            # When source moves right, image moves left (and vice versa)
            dist_from_wall = base_source_x - wall_x
            source_x = f"({base_source_x} + {amplitude * 0.3} * sin({omega} * t))"
            source_y = cy
            # Image is reflection: image_x = 2*wall_x - source_x
            image_x = f"({2 * wall_x} - ({base_source_x} + {amplitude * 0.3} * sin({omega} * t)))"
            image_y = cy

            gaussian_source = f"exp(-((x - {source_x})**2 + (y - {source_y})**2) / (2 * {sigma}**2))"
            gaussian_image = f"exp(-((x - {image_x})**2 + (y - {image_y})**2) / (2 * {sigma}**2))"

        elif motion == "circular":
            # Source orbits in a circle in the right half
            # Center of orbit is between wall and source base position
            orbit_cx = (wall_x + base_source_x) / 2 + amplitude * 0.3
            orbit_cy = cy
            orbit_r = min(amplitude, (base_source_x - wall_x) * 0.4)

            source_x = f"({orbit_cx} + {orbit_r} * cos({omega} * t))"
            source_y = f"({orbit_cy} + {orbit_r} * sin({omega} * t))"

            # Image reflects about wall
            image_x = f"({2 * wall_x} - ({orbit_cx} + {orbit_r} * cos({omega} * t)))"
            image_y = f"({orbit_cy} + {orbit_r} * sin({omega} * t))"

            gaussian_source = f"exp(-((x - {source_x})**2 + (y - {source_y})**2) / (2 * {sigma}**2))"
            gaussian_image = f"exp(-((x - {image_x})**2 + (y - {image_y})**2) / (2 * {sigma}**2))"

        elif motion == "diagonal":
            # Source moves diagonally (combination of x and y motion)
            source_x = f"({base_source_x} + {amplitude * 0.2} * sin({omega} * t))"
            source_y = f"({cy} + {amplitude} * cos({omega} * t))"
            image_x = f"({2 * wall_x} - ({base_source_x} + {amplitude * 0.2} * sin({omega} * t)))"
            image_y = f"({cy} + {amplitude} * cos({omega} * t))"

            gaussian_source = f"exp(-((x - {source_x})**2 + (y - {source_y})**2) / (2 * {sigma}**2))"
            gaussian_image = f"exp(-((x - {image_x})**2 + (y - {image_y})**2) / (2 * {sigma}**2))"

        else:
            # Default to static
            return PDE(
                rhs={
                    "phi": f"laplace(phi) - {strength} * s",
                    "s": "0",
                },
                **self._get_pde_bc_kwargs(bc),
            )

        # Both source and image are sources (same sign) to create no-flux at wall
        forcing = f"{strength * norm} * ({gaussian_source} + {gaussian_image})"

        return PDE(
            rhs={
                "phi": f"laplace(phi) - {forcing}",
                "s": "0",  # Not used for moving sources but kept for field structure
            },
            **self._get_pde_bc_kwargs(bc),
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
        motion = ic_params.get("motion", "static")

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
            # Using 2*pi for 2D potential (not 4*pi which is for 3D)
            phi_data = -(strength / (2 * np.pi)) * (
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

            phi_data = -(strength / (2 * np.pi)) * np.log(r_source_sq)

            radius = dx * 3
            s_data = np.exp(-r_source_sq / (2 * radius**2))

        elif motion in ("parallel", "perpendicular", "circular", "diagonal"):
            # For moving sources, start with zero potential
            # The field will develop structure as sources move
            phi_data = np.zeros(grid.shape)
            s_data = np.zeros(grid.shape)

        else:
            # Default: zero potential
            phi_data = np.zeros(grid.shape)
            s_data = np.zeros(grid.shape)

        phi = ScalarField(grid, phi_data)
        phi.label = "phi"
        s = ScalarField(grid, s_data)
        s.label = "s"

        return FieldCollection([phi, s])

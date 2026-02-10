"""Potential flow dipoles - source-sink pair dynamics with optional motion."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("potential-flow-dipoles")
class PotentialFlowDipolesPDE(MultiFieldPDEPreset):
    """Dipole (doublet) flows in 2D potential flow with moving sources.

    In potential flow theory, singularity solutions represent idealized point
    forces, vortices, sources, and sinks. A dipole arises when two equal and
    opposite singularities are brought infinitesimally close together.

    This implementation supports both static and moving source-sink pairs:
    - Static: Sources remain fixed, system relaxes to steady state
    - Circular: Dipole rotates around domain center
    - Oscillating: Dipole oscillates back and forth

    The simulation uses smooth Gaussian sources for numerical stability:
        d(phi)/dt = laplace(phi) + strength * (G_source - G_sink)

    where G_source and G_sink are Gaussians at time-dependent positions.

    Velocity from potential:
        u = d_dx(phi), v = d_dy(phi)

    Parameters:
        - strength: Source/sink intensity
        - separation: Distance between source and sink
        - sigma: Gaussian width for smooth sources
        - omega: Angular velocity for circular motion (rad/time)
        - motion: Type of motion (static, circular, oscillating)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="potential-flow-dipoles",
            category="fluids",
            description="Potential flow dipole with moving source-sink pair",
            equations={
                "phi": "laplace(phi) + strength * (G_source(t) - G_sink(t))",
            },
            parameters=[
                PDEParameter("strength", "Source/sink strength"),
                PDEParameter("separation", "Distance between source and sink"),
                PDEParameter("sigma", "Gaussian width for smooth sources"),
                PDEParameter("omega", "Angular velocity (rad/time) for motion"),
                PDEParameter("orbit_radius", "Radius of circular orbit around center"),
                PDEParameter("motion", "Motion type: static, circular, oscillating, figure8"),
            ],
            num_fields=1,
            field_names=["phi"],
            reference="Potential flow theory with time-dependent sources",
            supported_dimensions=[2],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
        **kwargs,
    ) -> PDE:
        """Create PDE with moving Gaussian sources."""
        strength = parameters["strength"]
        separation = parameters["separation"]
        sigma = parameters["sigma"]
        omega = parameters["omega"]
        orbit_radius = parameters["orbit_radius"]
        motion = parameters["motion"]

        # Get domain center
        x_min, x_max = grid.axes_bounds[0]
        y_min, y_max = grid.axes_bounds[1]
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        # Half separation for source/sink offset from dipole center
        half_sep = separation / 2

        if motion == "static":
            # Static sources at fixed positions
            # Source at (cx + half_sep, cy), sink at (cx - half_sep, cy)
            source_x = cx + half_sep
            source_y = cy
            sink_x = cx - half_sep
            sink_y = cy

            gaussian_source = f"exp(-((x - {source_x})**2 + (y - {source_y})**2) / (2 * {sigma}**2))"
            gaussian_sink = f"exp(-((x - {sink_x})**2 + (y - {sink_y})**2) / (2 * {sigma}**2))"

        elif motion == "circular":
            # Dipole center orbits around domain center
            # Dipole center position: (cx + R*cos(omega*t), cy + R*sin(omega*t))
            # Source/sink are offset from dipole center along the tangent direction
            dipole_cx = f"({cx} + {orbit_radius} * cos({omega} * t))"
            dipole_cy = f"({cy} + {orbit_radius} * sin({omega} * t))"

            # Tangent direction for source-sink alignment (perpendicular to radius)
            # tangent = (-sin(omega*t), cos(omega*t))
            source_x = f"({dipole_cx} - {half_sep} * sin({omega} * t))"
            source_y = f"({dipole_cy} + {half_sep} * cos({omega} * t))"
            sink_x = f"({dipole_cx} + {half_sep} * sin({omega} * t))"
            sink_y = f"({dipole_cy} - {half_sep} * cos({omega} * t))"

            gaussian_source = f"exp(-((x - {source_x})**2 + (y - {source_y})**2) / (2 * {sigma}**2))"
            gaussian_sink = f"exp(-((x - {sink_x})**2 + (y - {sink_y})**2) / (2 * {sigma}**2))"

        elif motion == "oscillating":
            # Dipole oscillates horizontally through center
            # Position: (cx + orbit_radius * sin(omega*t), cy)
            dipole_cx = f"({cx} + {orbit_radius} * sin({omega} * t))"
            dipole_cy = f"{cy}"

            # Source/sink aligned vertically (perpendicular to motion)
            source_x = dipole_cx
            source_y = f"({dipole_cy} + {half_sep})"
            sink_x = dipole_cx
            sink_y = f"({dipole_cy} - {half_sep})"

            gaussian_source = f"exp(-((x - {source_x})**2 + (y - {source_y})**2) / (2 * {sigma}**2))"
            gaussian_sink = f"exp(-((x - {sink_x})**2 + (y - {sink_y})**2) / (2 * {sigma}**2))"

        elif motion == "figure8":
            # Lissajous figure-8 pattern: x = A*sin(wt), y = A*sin(2wt)
            dipole_cx = f"({cx} + {orbit_radius} * sin({omega} * t))"
            dipole_cy = f"({cy} + {orbit_radius} * sin({2 * omega} * t))"

            source_x = f"({dipole_cx} + {half_sep})"
            source_y = dipole_cy
            sink_x = f"({dipole_cx} - {half_sep})"
            sink_y = dipole_cy

            gaussian_source = f"exp(-((x - {source_x})**2 + (y - {source_y})**2) / (2 * {sigma}**2))"
            gaussian_sink = f"exp(-((x - {sink_x})**2 + (y - {sink_y})**2) / (2 * {sigma}**2))"

        else:
            raise ValueError(f"Unknown motion type: {motion!r}. Must be one of: static, circular, oscillating, figure8")

        # Normalize Gaussians and compute forcing
        norm = 1.0 / (2 * np.pi * sigma**2)
        forcing = f"{strength * norm} * ({gaussian_source} - {gaussian_sink})"

        return PDE(
            rhs={
                "phi": f"laplace(phi) + {forcing}",
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
        """Create initial potential field.

        The potential starts at zero and evolves as sources move.
        """
        # Initialize phi to zero - it will develop structure as sources move
        phi_data = np.zeros(grid.shape)

        phi = ScalarField(grid, phi_data)
        phi.label = "phi"

        return FieldCollection([phi])

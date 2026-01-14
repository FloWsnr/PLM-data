"""2D Navier-Stokes equations with cylindrical obstacle for vortex shedding."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("navier-stokes-cylinder")
class NavierStokesCylinderPDE(MultiFieldPDEPreset):
    """2D Navier-Stokes with cylindrical obstacle for vortex shedding.

    This preset models flow around a cylindrical obstacle, which creates
    the famous von Kármán vortex street at appropriate Reynolds numbers.

    The obstacle is represented as a penalization/immersed boundary
    in the S field, with damping terms applied to the velocity.

    Momentum equations with obstacle damping:
        du/dt = -(u * d_dx(u) + v * d_dy(u)) - d_dx(p) + nu * laplace(u) - u * max(S, 0)
        dv/dt = -(u * d_dx(v) + v * d_dy(v)) - d_dy(p) + nu * laplace(v) - v * max(S, 0)

    Generalized pressure equation:
        dp/dt = nu * laplace(p) - (1/M^2) * (d_dx(u) + d_dy(v))

    Obstacle field (static):
        dS/dt = 0

    where:
        - u, v are velocity components
        - p is pressure
        - S is obstacle field (1 inside obstacle, 0 outside)
        - nu is kinematic viscosity
        - M is Mach number parameter
        - The term -u*max(S,0) creates strong damping inside obstacles

    Reference: Visual PDE NavierStokesFlowCylinder preset
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="navier-stokes-cylinder",
            category="fluids",
            description="Navier-Stokes flow around cylindrical obstacle (vortex shedding)",
            equations={
                "u": "-(u * d_dx(u) + v * d_dy(u)) - d_dx(p) + nu * laplace(u) - u * max(S, 0)",
                "v": "-(u * d_dx(v) + v * d_dy(v)) - d_dy(p) + nu * laplace(v) - v * max(S, 0)",
                "p": "nu * laplace(p) - (1/M^2) * (d_dx(u) + d_dy(v))",
                "S": "0 (static obstacle field)",
            },
            parameters=[
                PDEParameter(
                    name="nu",
                    default=0.1,
                    description="Kinematic viscosity",
                    min_value=0.001,
                    max_value=20.0,
                ),
                PDEParameter(
                    name="M",
                    default=0.5,
                    description="Mach number parameter",
                    min_value=0.1,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="U",
                    default=0.7,
                    description="Inflow velocity",
                    min_value=0.1,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="cylinder_radius",
                    default=0.05,
                    description="Cylinder radius (fraction of domain)",
                    min_value=0.01,
                    max_value=0.2,
                ),
            ],
            num_fields=4,
            field_names=["u", "v", "p", "S"],
            reference="Von Kármán vortex street (1911)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        nu = parameters.get("nu", 0.1)
        M = parameters.get("M", 0.5)

        # Pressure relaxation coefficient
        inv_M2 = 1.0 / (M * M)

        # The max(S, 0) term ensures damping only where S > 0 (inside obstacle)
        # We use Heaviside(S) * S which for positive S gives S (the damping coefficient)
        # In py-pde, we can approximate max(S, 0) as 0.5 * (S + abs(S))
        # Or simply use S since we initialize S >= 0 everywhere

        return PDE(
            rhs={
                "u": f"-u * d_dx(u) - v * d_dy(u) - d_dx(p) + {nu} * laplace(u) - u * S",
                "v": f"-u * d_dx(v) - v * d_dy(v) - d_dy(p) + {nu} * laplace(v) - v * S",
                "p": f"{nu} * laplace(p) - {inv_M2} * (d_dx(u) + d_dy(v))",
                "S": "0",  # Static obstacle field
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
        """Create initial flow around cylinder.

        Initial conditions:
        - u = U (uniform flow in positive x direction)
        - v = 0 (no vertical velocity)
        - p = 0 (uniform pressure)
        - S = 1 inside cylinder, 0 outside (obstacle indicator)
        """
        # Get domain bounds from grid
        x_min, x_max = grid.axes_bounds[0]
        y_min, y_max = grid.axes_bounds[1]
        L_x = x_max - x_min
        L_y = y_max - y_min

        # Get parameters
        U = ic_params.get("U", 0.7)
        cylinder_radius = ic_params.get("cylinder_radius", 0.05)
        cx = ic_params.get("cylinder_x", 0.25)  # Cylinder center x (fraction)
        cy = ic_params.get("cylinder_y", 0.5)   # Cylinder center y (fraction)

        x, y = np.meshgrid(
            np.linspace(x_min, x_max, grid.shape[0]),
            np.linspace(y_min, y_max, grid.shape[1]),
            indexing="ij",
        )

        # Normalize coordinates
        x_norm = (x - x_min) / L_x
        y_norm = (y - y_min) / L_y

        if ic_type in ("cylinder-flow", "default", "navier-stokes-cylinder-default"):
            # Uniform inflow velocity
            u_data = np.full_like(x, U)
            v_data = np.zeros_like(x)
            p_data = np.zeros_like(x)

            # Cylinder obstacle: S = 1 inside, 0 outside
            # Use smooth transition for numerical stability
            r_sq = (x_norm - cx) ** 2 + (y_norm - cy) ** 2
            radius_sq = cylinder_radius**2

            # Smooth indicator: 1 inside cylinder, 0 outside
            # Using tanh for smooth transition
            sharpness = 100.0  # Higher = sharper boundary
            S_data = 0.5 * (1.0 - np.tanh(sharpness * (r_sq - radius_sq)))

            # Zero velocity inside cylinder
            u_data = u_data * (1.0 - S_data)
            v_data = v_data * (1.0 - S_data)

        elif ic_type == "multi-cylinder":
            # Multiple cylinders for complex vortex interactions
            n_cylinders = int(ic_params.get("n_cylinders", 2))
            positions = ic_params.get(
                "positions", [(0.25, 0.35), (0.25, 0.65)]
            )

            u_data = np.full_like(x, U)
            v_data = np.zeros_like(x)
            p_data = np.zeros_like(x)
            S_data = np.zeros_like(x)

            sharpness = 100.0
            for i in range(min(n_cylinders, len(positions))):
                cx_i, cy_i = positions[i]
                r_sq = (x_norm - cx_i) ** 2 + (y_norm - cy_i) ** 2
                radius_sq = cylinder_radius**2
                S_i = 0.5 * (1.0 - np.tanh(sharpness * (r_sq - radius_sq)))
                S_data = np.maximum(S_data, S_i)

            # Zero velocity inside cylinders
            u_data = u_data * (1.0 - S_data)
            v_data = v_data * (1.0 - S_data)

        else:
            # Default to single cylinder
            u_data = np.full_like(x, U)
            v_data = np.zeros_like(x)
            p_data = np.zeros_like(x)

            r_sq = (x_norm - cx) ** 2 + (y_norm - cy) ** 2
            radius_sq = cylinder_radius**2
            sharpness = 100.0
            S_data = 0.5 * (1.0 - np.tanh(sharpness * (r_sq - radius_sq)))

            u_data = u_data * (1.0 - S_data)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"
        p = ScalarField(grid, p_data)
        p.label = "p"
        S = ScalarField(grid, S_data)
        S.label = "S"

        return FieldCollection([u, v, p, S])

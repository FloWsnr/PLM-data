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
                PDEParameter("nu", "Kinematic viscosity"),
                PDEParameter("M", "Mach number parameter"),
                PDEParameter("U", "Inflow velocity"),
                PDEParameter("cylinder_radius", "Cylinder radius (fraction of domain)"),
            ],
            num_fields=4,
            field_names=["u", "v", "p", "S"],
            reference="Von Kármán vortex street (1911)",
            supported_dimensions=[2],
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

        x, y = np.meshgrid(
            np.linspace(x_min, x_max, grid.shape[0]),
            np.linspace(y_min, y_max, grid.shape[1]),
            indexing="ij",
        )

        # Normalize coordinates
        x_norm = (x - x_min) / L_x
        y_norm = (y - y_min) / L_y

        randomize = kwargs.get("randomize", False)

        if ic_type in ("cylinder-flow", "default", "navier-stokes-cylinder-default"):
            cx = ic_params.get("cylinder_x")
            cy = ic_params.get("cylinder_y")
            if randomize:
                rng = np.random.default_rng(ic_params.get("seed"))
                cx = rng.uniform(0.0, 1.0)
                cy = rng.uniform(0.0, 1.0)
            if cx is None or cy is None:
                raise ValueError("navier-stokes-cylinder requires cylinder_x and cylinder_y")
            # Uniform inflow velocity everywhere (including inside cylinder)
            # The damping term -u*S will naturally bring velocity to zero inside
            u_data = np.full_like(x, U)
            v_data = np.zeros_like(x)
            p_data = np.zeros_like(x)

            # Cylinder obstacle: S = 1 inside, 0 outside (binary like reference)
            r_sq = (x_norm - cx) ** 2 + (y_norm - cy) ** 2
            radius_sq = cylinder_radius**2

            # Binary indicator: exactly 1 inside cylinder, 0 outside (matches reference)
            S_data = np.where(r_sq < radius_sq, 1.0, 0.0)

            # Don't zero velocity - let damping term handle it naturally
            # This matches the reference implementation

        elif ic_type == "multi-cylinder":
            # Multiple cylinders for complex vortex interactions
            positions = ic_params.get("positions")
            if randomize:
                if "n_cylinders" not in ic_params:
                    raise ValueError("navier-stokes-cylinder multi-cylinder random positions require n_cylinders")
                rng = np.random.default_rng(ic_params.get("seed"))
                positions = [
                    [rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)]
                    for _ in range(int(ic_params["n_cylinders"]))
                ]
            if positions is None:
                raise ValueError("navier-stokes-cylinder multi-cylinder requires positions")
            n_cylinders = int(ic_params.get("n_cylinders")) if "n_cylinders" in ic_params else len(positions)
            if n_cylinders > len(positions):
                raise ValueError("navier-stokes-cylinder multi-cylinder positions length must match n_cylinders")

            u_data = np.full_like(x, U)
            v_data = np.zeros_like(x)
            p_data = np.zeros_like(x)
            S_data = np.zeros_like(x)

            # Binary indicator for each cylinder
            for i in range(min(n_cylinders, len(positions))):
                cx_i, cy_i = positions[i]
                r_sq = (x_norm - cx_i) ** 2 + (y_norm - cy_i) ** 2
                radius_sq = cylinder_radius**2
                S_i = np.where(r_sq < radius_sq, 1.0, 0.0)
                S_data = np.maximum(S_data, S_i)

            # Don't zero velocity - let damping handle it

        else:
            # Default to single cylinder
            cx = ic_params.get("cylinder_x")
            cy = ic_params.get("cylinder_y")
            if randomize:
                rng = np.random.default_rng(ic_params.get("seed"))
                cx = rng.uniform(0.0, 1.0)
                cy = rng.uniform(0.0, 1.0)
            if cx is None or cy is None:
                raise ValueError("navier-stokes-cylinder requires cylinder_x and cylinder_y")
            u_data = np.full_like(x, U)
            v_data = np.zeros_like(x)
            p_data = np.zeros_like(x)

            r_sq = (x_norm - cx) ** 2 + (y_norm - cy) ** 2
            radius_sq = cylinder_radius**2
            # Binary indicator: exactly 1 inside cylinder, 0 outside
            S_data = np.where(r_sq < radius_sq, 1.0, 0.0)

            # Don't zero velocity - let damping handle it

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"
        p = ScalarField(grid, p_data)
        p.label = "p"
        S = ScalarField(grid, S_data)
        S.label = "S"

        return FieldCollection([u, v, p, S])


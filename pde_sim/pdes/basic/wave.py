"""Wave equation and variants."""

import math
from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("wave")
class WavePDE(MultiFieldPDEPreset):
    """Wave equation.

    The wave equation describes wave propagation:

        d^2u/dt^2 = D * laplace(u)

    Based on visualpde.com, converted to first order system with stabilization:
        du/dt = v + C*D * laplace(u)
        dv/dt = D * laplace(u)

    where:
        - u is displacement
        - v is velocity (du/dt)
        - D is the wave speed squared (c^2)
        - C is a damping/stabilization parameter (0 for pure wave equation)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="wave",
            category="basic",
            description="Wave equation (second order hyperbolic PDE)",
            equations={
                "u": "v + C*D * laplace(u)",
                "v": "D * laplace(u)",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Wave speed squared (c^2)",
                    min_value=1.0,
                    max_value=100.0,
                ),
                PDEParameter(
                    name="C",
                    default=0.01,
                    description="Damping/stabilization coefficient",
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="https://visualpde.com/basic-pdes/wave-equation",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        C = parameters.get("C", 0.01)

        bc_spec = self._convert_bc(bc)

        # Build u equation with optional damping term
        if C > 0:
            u_rhs = f"v + {C * D} * laplace(u)"
        else:
            u_rhs = "v"

        return PDE(
            rhs={
                "u": u_rhs,
                "v": f"{D} * laplace(u)",
            },
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial state for wave equation."""
        # u gets the specified initial condition
        u = create_initial_condition(grid, ic_type, ic_params)
        u.label = "u"

        # v (velocity) starts at zero by default
        v_data = np.zeros(grid.shape)
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])


@register_pde("inhomogeneous-wave")
class InhomogeneousWavePDE(MultiFieldPDEPreset):
    """Inhomogeneous wave equation with spatially varying wave speed.

    Based on the visualpde.com inhomogeneous wave equation:

        d^2u/dt^2 = div(f(x,y) * grad(u))

    where f(x,y) is a spatially varying diffusivity:

        f(x,y) = D * [1 + E*sin(m*pi*x/L_x)] * [1 + E*sin(n*pi*y/L_y)]

    This creates a wave equation with position-dependent wave speed,
    leading to interesting refraction and focusing effects.

    Converted to first-order system:
        du/dt = v + C*D * laplace(u)
        dv/dt = div(f(x,y) * grad(u))

    The divergence of (f * grad(u)) expands to:
        f * laplace(u) + grad(f) . grad(u)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="inhomogeneous-wave",
            category="basic",
            description="Wave equation with spatially varying wave speed",
            equations={
                "u": "v + C*D * laplace(u)",
                "v": "div(f(x,y) * grad(u))",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Base diffusivity (wave speed squared)",
                    min_value=0.1,
                    max_value=100.0,
                ),
                PDEParameter(
                    name="E",
                    default=0.97,
                    description="Amplitude of spatial variation (|E| < 1)",
                    min_value=-0.99,
                    max_value=0.99,
                ),
                PDEParameter(
                    name="m",
                    default=9,
                    description="Spatial mode number in x direction",
                    min_value=0,
                    max_value=10,
                ),
                PDEParameter(
                    name="n",
                    default=9,
                    description="Spatial mode number in y direction",
                    min_value=0,
                    max_value=10,
                ),
                PDEParameter(
                    name="C",
                    default=0.01,
                    description="Damping/stabilization coefficient",
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="https://visualpde.com/basic-pdes/inhomogeneous-wave-equation",
            supported_dimensions=[2],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        E = parameters.get("E", 0.97)
        m = int(parameters.get("m", 9))
        n = int(parameters.get("n", 9))
        C = parameters.get("C", 0.01)

        # Get domain size from grid
        L_x = grid.axes_bounds[0][1] - grid.axes_bounds[0][0]
        L_y = grid.axes_bounds[1][1] - grid.axes_bounds[1][0]

        # Get spatial coordinates
        x_coords = grid.cell_coords[..., 0]
        y_coords = grid.cell_coords[..., 1]

        # Compute f(x,y) = D * [1 + E*sin(m*pi*x/L_x)] * [1 + E*sin(n*pi*y/L_y)]
        sin_x = np.sin(m * math.pi * x_coords / L_x)
        sin_y = np.sin(n * math.pi * y_coords / L_y)
        f_data = D * (1 + E * sin_x) * (1 + E * sin_y)
        f_field = ScalarField(grid, f_data)

        # Compute df/dx = D * E * (m*pi/L_x) * cos(m*pi*x/L_x) * [1 + E*sin(n*pi*y/L_y)]
        cos_x = np.cos(m * math.pi * x_coords / L_x)
        df_dx_data = D * E * (m * math.pi / L_x) * cos_x * (1 + E * sin_y)
        df_dx_field = ScalarField(grid, df_dx_data)

        # Compute df/dy = D * E * (n*pi/L_y) * cos(n*pi*y/L_y) * [1 + E*sin(m*pi*x/L_x)]
        cos_y = np.cos(n * math.pi * y_coords / L_y)
        df_dy_data = D * E * (n * math.pi / L_y) * cos_y * (1 + E * sin_x)
        df_dy_field = ScalarField(grid, df_dy_data)

        # div(f * grad(u)) = f * laplace(u) + df_dx * d_dx(u) + df_dy * d_dy(u)
        bc_spec = self._convert_bc(bc)

        # Build u equation with optional damping term
        if C > 0:
            u_rhs = f"v + {C * D} * laplace(u)"
        else:
            u_rhs = "v"

        return PDE(
            rhs={
                "u": u_rhs,
                "v": "f * laplace(u) + df_dx * d_dx(u) + df_dy * d_dy(u)",
            },
            bc=bc_spec,
            consts={
                "f": f_field,
                "df_dx": df_dx_field,
                "df_dy": df_dy_field,
            },
        )

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        D = parameters.get("D", 1.0)
        E = parameters.get("E", 0.97)
        m = int(parameters.get("m", 9))
        n = int(parameters.get("n", 9))
        C = parameters.get("C", 0.01)
        return {
            "u": f"v + {C}*{D} * laplace(u)" if C > 0 else "v",
            "v": "div(f(x,y) * grad(u))",
            "f(x,y)": f"D*[1+E*sin(m*pi*x/L_x)]*[1+E*sin(n*pi*y/L_y)] with D={D}, E={E}, m={m}, n={n}",
        }

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial state for inhomogeneous wave equation."""
        # u gets the specified initial condition
        u = create_initial_condition(grid, ic_type, ic_params)
        u.label = "u"

        # v (velocity) starts at zero by default
        v_data = np.zeros(grid.shape)
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

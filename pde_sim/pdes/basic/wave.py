"""Wave equation."""

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

        d²u/dt² = c² * laplace(u)

    We convert to first order system:
        du/dt = v
        dv/dt = c² * laplace(u)

    where u is displacement and v is velocity.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="wave",
            category="basic",
            description="Wave equation (second order)",
            equations={
                "u": "v",
                "v": "c**2 * laplace(u)",
            },
            parameters=[
                PDEParameter(
                    name="c",
                    default=1.0,
                    description="Wave speed",
                    min_value=0.5,
                    max_value=2.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Hyperbolic PDE",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        c = parameters.get("c", 1.0)
        c_sq = c * c

        bc_spec = self._convert_bc(bc)
        return PDE(
            rhs={
                "u": "v",
                "v": f"{c_sq} * laplace(u)",
            },
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
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


@register_pde("advection")
class AdvectionPDE(MultiFieldPDEPreset):
    """Advection-diffusion equation.

    Describes transport of a quantity by flow and diffusion:

        du/dt = D * laplace(u) - v · grad(u)

    where D is diffusion and v is velocity vector.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="advection",
            category="basic",
            description="Advection-diffusion equation",
            equations={
                "u": "D * laplace(u) - vx * d_dx(u) - vy * d_dy(u)",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.05,
                    description="Diffusion coefficient",
                    min_value=0.0,
                    max_value=0.2,
                ),
                PDEParameter(
                    name="vx",
                    default=0.5,
                    description="Advection velocity in x",
                    min_value=-2.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="vy",
                    default=0.0,
                    description="Advection velocity in y",
                    min_value=-2.0,
                    max_value=2.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Convection-diffusion equation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 0.01)
        vx = parameters.get("vx", 1.0)
        vy = parameters.get("vy", 0.0)

        # Build equation string
        terms = []
        if D > 0:
            terms.append(f"{D} * laplace(u)")
        if vx != 0:
            terms.append(f"- {vx} * d_dx(u)")
        if vy != 0:
            terms.append(f"- {vy} * d_dy(u)")

        rhs = " ".join(terms) if terms else "0"

        bc_spec = self._convert_bc(bc)
        return PDE(
            rhs={"u": rhs},
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        return create_initial_condition(grid, ic_type, ic_params)


@register_pde("inhomogeneous-wave")
class InhomogeneousWavePDE(MultiFieldPDEPreset):
    """Inhomogeneous wave equation with spatially varying wave speed.

    Based on the visualpde.com inhomogeneous wave equation:

        d²u/dt² = div(f(x,y) * grad(u))

    where f(x,y) is a spatially varying diffusivity:

        f(x,y) = D * [1 + E*sin(m*pi*x/L)] * [1 + E*sin(n*pi*y/L)]

    This creates a wave equation with position-dependent wave speed,
    leading to interesting refraction and focusing effects.

    Converted to first-order system:
        du/dt = v
        dv/dt = div(f(x,y) * grad(u))

    The divergence of (f * grad(u)) expands to:
        f * laplace(u) + grad(f) · grad(u)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="inhomogeneous-wave",
            category="basic",
            description="Wave equation with spatially varying wave speed",
            equations={
                "u": "v",
                "v": "div(f(x,y) * grad(u))",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Base diffusivity (wave speed squared)",
                    min_value=0.1,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="E",
                    default=0.5,
                    description="Amplitude of spatial variation (|E| < 1)",
                    min_value=-0.9,
                    max_value=0.9,
                ),
                PDEParameter(
                    name="n",
                    default=2,
                    description="Spatial mode number in x direction",
                    min_value=1,
                    max_value=5,
                ),
                PDEParameter(
                    name="m",
                    default=2,
                    description="Spatial mode number in y direction",
                    min_value=1,
                    max_value=5,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="https://visualpde.com/basic-pdes/inhomogeneous-wave-equation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        E = parameters.get("E", 0.5)
        n = int(parameters.get("n", 2))
        m = int(parameters.get("m", 2))

        # Get domain size from grid
        L = grid.axes_bounds[0][1] - grid.axes_bounds[0][0]

        # Get spatial coordinates
        x_coords = grid.cell_coords[..., 0]
        y_coords = grid.cell_coords[..., 1]

        # Compute f(x,y) = D * [1 + E*sin(m*pi*x/L)] * [1 + E*sin(n*pi*y/L)]
        sin_x = np.sin(m * math.pi * x_coords / L)
        sin_y = np.sin(n * math.pi * y_coords / L)
        f_data = D * (1 + E * sin_x) * (1 + E * sin_y)
        f_field = ScalarField(grid, f_data)

        # Compute df/dx = D * E * (m*pi/L) * cos(m*pi*x/L) * [1 + E*sin(n*pi*y/L)]
        cos_x = np.cos(m * math.pi * x_coords / L)
        df_dx_data = D * E * (m * math.pi / L) * cos_x * (1 + E * sin_y)
        df_dx_field = ScalarField(grid, df_dx_data)

        # Compute df/dy = D * E * (n*pi/L) * cos(n*pi*y/L) * [1 + E*sin(m*pi*x/L)]
        cos_y = np.cos(n * math.pi * y_coords / L)
        df_dy_data = D * E * (n * math.pi / L) * cos_y * (1 + E * sin_x)
        df_dy_field = ScalarField(grid, df_dy_data)

        # div(f * grad(u)) = f * laplace(u) + df_dx * d_dx(u) + df_dy * d_dy(u)
        bc_spec = self._convert_bc(bc)
        return PDE(
            rhs={
                "u": "v",
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
        E = parameters.get("E", 0.5)
        n = int(parameters.get("n", 2))
        m = int(parameters.get("m", 2))
        return {
            "u": "v",
            "v": "div(f(x,y) * grad(u))",
            "f(x,y)": f"D*[1+E*sin(mπx/L)]*[1+E*sin(nπy/L)] with D={D}, E={E}, n={n}, m={m}",
        }

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
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

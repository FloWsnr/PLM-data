"""Complex Ginzburg-Landau equation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import PDEMetadata, PDEParameter, MultiFieldPDEPreset
from .. import register_pde


@register_pde("complex-ginzburg-landau")
class ComplexGinzburgLandauPDE(MultiFieldPDEPreset):
    """Complex Ginzburg-Landau equation.

    Based on visualpde.com formulation:

        dpsi/dt = (D_r + i*D_i)*laplace(psi) + (a_r + i*a_i)*psi + (b_r + i*b_i)*psi*|psi|^2

    Writing psi = u + i*v, this becomes:
        du/dt = D_r*laplace(u) - D_i*laplace(v) + a_r*u - a_i*v + (b_r*u - b_i*v)*(u^2 + v^2)
        dv/dt = D_i*laplace(u) + D_r*laplace(v) + a_r*v + a_i*u + (b_r*v + b_i*u)*(u^2 + v^2)

    One of the most universal equations in nonlinear physics, describing
    amplitude dynamics near oscillatory instabilities.

    Key phenomena:
        - Spiral waves: self-organized rotating patterns
        - Defect turbulence: chaotic dynamics mediated by topological defects
        - Plane waves and traveling waves
        - Phase turbulence: disorder in phase with constant amplitude

    The dynamics depend on Benjamin-Feir stability: 1 + c1*c3 > 0 for stable plane waves,
    where c1 = D_i/D_r and c3 = b_i/|b_r|.

    Reference: Ginzburg & Landau (1950), Aranson & Kramer (2002)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="complex-ginzburg-landau",
            category="physics",
            description="Complex Ginzburg-Landau for spiral waves and turbulence",
            equations={
                "u": "D_r*laplace(u) - D_i*laplace(v) + a_r*u - a_i*v + (b_r*u - b_i*v)*(u**2 + v**2)",
                "v": "D_i*laplace(u) + D_r*laplace(v) + a_r*v + a_i*u + (b_r*v + b_i*u)*(u**2 + v**2)",
            },
            parameters=[
                PDEParameter("D_r", "Real diffusion coefficient"),
                PDEParameter("D_i", "Imaginary diffusion (dispersion)"),
                PDEParameter("a_r", "Linear growth rate"),
                PDEParameter("a_i", "Linear frequency"),
                PDEParameter("b_r", "Nonlinear saturation (must be negative)"),
                PDEParameter("b_i", "Nonlinear frequency shift"),
                PDEParameter("n", "Initial condition mode number (x)"),
                PDEParameter("m", "Initial condition mode number (y)"),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Aranson & Kramer (2002) World of the CGL equation",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Complex Ginzburg-Landau PDE.

        Args:
            parameters: Dictionary with D_r, D_i, a_r, a_i, b_r, b_i.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        D_r = parameters.get("D_r", 0.1)
        D_i = parameters.get("D_i", 1.0)
        a_r = parameters.get("a_r", 1.0)
        a_i = parameters.get("a_i", 1.0)
        b_r = parameters.get("b_r", -1.0)
        b_i = parameters.get("b_i", 0.0)

        # Complex Ginzburg-Landau in real/imaginary form:
        # du/dt = D_r*laplace(u) - D_i*laplace(v) + a_r*u - a_i*v + (b_r*u - b_i*v)*(u^2 + v^2)
        # dv/dt = D_i*laplace(u) + D_r*laplace(v) + a_r*v + a_i*u + (b_r*v + b_i*u)*(u^2 + v^2)
        return PDE(
            rhs={
                "u": f"{D_r}*laplace(u) - {D_i}*laplace(v) + {a_r}*u - {a_i}*v + ({b_r}*u - {b_i}*v)*(u**2 + v**2)",
                "v": f"{D_i}*laplace(u) + {D_r}*laplace(v) + {a_r}*v + {a_i}*u + ({b_r}*v + {b_i}*u)*(u**2 + v**2)",
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
        """Create initial state for CGL.

        Default: sinusoidal modes in x and y.
        """
        if ic_type in ("complex-ginzburg-landau-default", "default"):
            n = int(ic_params.get("n", 10))
            m = int(ic_params.get("m", 10))
            amplitude = ic_params.get("amplitude", 1.0)
            rng = np.random.default_rng(ic_params.get("seed"))

            # Get domain info
            x_bounds = grid.axes_bounds[0]
            y_bounds = grid.axes_bounds[1]
            Lx = x_bounds[1] - x_bounds[0]
            Ly = y_bounds[1] - y_bounds[0]

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            # Randomize phases if not specified
            phase_x = ic_params.get("phase_x")
            phase_y = ic_params.get("phase_y")
            if phase_x is None or phase_x == "random" or phase_y is None or phase_y == "random":
                raise ValueError("complex-ginzburg-landau requires phase_x and phase_y (or random)")

            # Sinusoidal initial condition: sin(n*pi*x/Lx + phase) * sin(m*pi*y/Ly + phase)
            u_data = amplitude * np.sin(n * np.pi * (X - x_bounds[0]) / Lx + phase_x) * np.sin(m * np.pi * (Y - y_bounds[0]) / Ly + phase_y)
            v_data = amplitude * np.sin(n * np.pi * (X - x_bounds[0]) / Lx + phase_x) * np.sin(m * np.pi * (Y - y_bounds[0]) / Ly + phase_y)

            # Add small noise
            u_data += 0.01 * rng.standard_normal(grid.shape)
            v_data += 0.01 * rng.standard_normal(grid.shape)

            u = ScalarField(grid, u_data)
            u.label = "u"
            v = ScalarField(grid, v_data)
            v.label = "v"

            return FieldCollection([u, v])

        # For other IC types, create the same IC for both fields
        base = create_initial_condition(grid, ic_type, ic_params)
        u = ScalarField(grid, base.data.copy())
        u.label = "u"
        v = ScalarField(grid, np.zeros(grid.shape))
        v.label = "v"
        return FieldCollection([u, v])

    def resolve_ic_params(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> dict[str, Any]:
        if ic_type in ("complex-ginzburg-landau-default", "default"):
            resolved = ic_params.copy()
            if "phase_x" not in resolved or "phase_y" not in resolved:
                raise ValueError("complex-ginzburg-landau requires phase_x and phase_y (or random)")
            if resolved["phase_x"] == "random" or resolved["phase_y"] == "random":
                rng = np.random.default_rng(resolved.get("seed"))
                if resolved["phase_x"] == "random":
                    resolved["phase_x"] = rng.uniform(0, 2 * np.pi)
                if resolved["phase_y"] == "random":
                    resolved["phase_y"] = rng.uniform(0, 2 * np.pi)
            if resolved["phase_x"] is None or resolved["phase_y"] is None:
                raise ValueError("complex-ginzburg-landau requires phase_x and phase_y (or random)")
            return resolved
        return super().resolve_ic_params(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        D_r = parameters.get("D_r", 0.1)
        D_i = parameters.get("D_i", 1.0)
        a_r = parameters.get("a_r", 1.0)
        a_i = parameters.get("a_i", 1.0)
        b_r = parameters.get("b_r", -1.0)
        b_i = parameters.get("b_i", 0.0)

        return {
            "u": f"{D_r}*laplace(u) - {D_i}*laplace(v) + {a_r}*u - {a_i}*v + ({b_r}*u - {b_i}*v)*(u**2 + v**2)",
            "v": f"{D_i}*laplace(u) + {D_r}*laplace(v) + {a_r}*v + {a_i}*u + ({b_r}*v + {b_i}*u)*(u**2 + v**2)",
        }

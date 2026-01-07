"""Nonlinear beam equation with state-dependent stiffness."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("nonlinear-beam")
class NonlinearBeamPDE(ScalarPDEPreset):
    """Overdamped nonlinear beam equation with state-dependent stiffness.

    Based on visualpde.com formulation (adapted to 2D):

        du/dt = -laplace(E(u) * laplace(u))

    where the stiffness E depends on local curvature:

        E(u) = E_star + Delta_E * (1 + tanh(laplace(u)/eps)) / 2

    For large curvatures (|laplace(u)| >> eps):
        - Positive curvature: E approx E_star + Delta_E (stiffer)
        - Negative curvature: E approx E_star (softer)

    Key features:
        - Material nonlinearity: stiffness varies with local bending
        - Curvature softening/hardening: controlled by Delta_E sign
        - Pattern formation: differential stiffness leads to complex dynamics
        - Localization: energy focuses in softer regions

    Applications:
        - MEMS resonators with nonlinear stiffness
        - Composite materials with varying properties
        - Biological structures (tendons, cartilage)
        - Smart materials (piezoelectrics, shape-memory alloys)

    Reference: Euler-Bernoulli beam theory, Nayfeh & Pai (2004)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="nonlinear-beam",
            category="physics",
            description="Overdamped beam with curvature-dependent stiffness",
            equations={
                "u": "-laplace(E(u) * laplace(u))",
                "E(u)": "E_star + Delta_E * (1 + tanh(laplace(u)/eps)) / 2",
            },
            parameters=[
                PDEParameter(
                    name="E_star",
                    default=0.0001,
                    description="Baseline stiffness scale",
                    min_value=0.0001,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="Delta_E",
                    default=24.0,
                    description="Stiffness variation range",
                    min_value=0.0,
                    max_value=50.0,
                ),
                PDEParameter(
                    name="eps",
                    default=0.01,
                    description="Stiffness transition sharpness",
                    min_value=0.001,
                    max_value=1.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Euler-Bernoulli beam theory",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Nonlinear Beam PDE.

        Args:
            parameters: Dictionary with E_star, Delta_E, eps.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        E_star = parameters.get("E_star", 0.0001)
        Delta_E = parameters.get("Delta_E", 24.0)
        eps = parameters.get("eps", 0.01)

        # Overdamped beam with state-dependent stiffness:
        # du/dt = -laplace(E(u) * laplace(u))
        # where E(u) = E_star + Delta_E * (1 + tanh(laplace(u)/eps)) / 2
        #
        # We approximate by making E depend on local curvature:
        # E_avg = E_star + Delta_E/2
        # E_var = Delta_E/2
        # du/dt = -(E_avg + E_var * tanh(laplace(u)/eps)) * laplace(laplace(u))

        E_avg = E_star + Delta_E / 2
        E_var = Delta_E / 2

        return PDE(
            rhs={
                "u": f"-({E_avg} + {E_var} * tanh(laplace(u) / {eps})) * laplace(laplace(u))"
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> ScalarField:
        """Create initial beam displacement.

        Default: sinusoidal initial shape with small noise.
        """
        if ic_type in ("nonlinear-beam-default", "default"):
            # Create a smooth initial displacement
            seed = ic_params.get("seed")
            if seed is not None:
                np.random.seed(seed)
            amplitude = ic_params.get("amplitude", 0.5)

            x_bounds = grid.axes_bounds[0]
            y_bounds = grid.axes_bounds[1]
            Lx = x_bounds[1] - x_bounds[0]
            Ly = y_bounds[1] - y_bounds[0]

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            # Sinusoidal initial shape
            data = amplitude * np.sin(np.pi * (X - x_bounds[0]) / Lx) * np.sin(np.pi * (Y - y_bounds[0]) / Ly)
            data += 0.05 * np.random.randn(*grid.shape)

            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        E_star = parameters.get("E_star", 0.0001)
        Delta_E = parameters.get("Delta_E", 24.0)
        eps = parameters.get("eps", 0.01)
        E_avg = E_star + Delta_E / 2
        E_var = Delta_E / 2

        return {
            "u": f"-({E_avg} + {E_var} * tanh(laplace(u) / {eps})) * laplace(laplace(u))"
        }

"""Nonlinear beam equation with state-dependent stiffness."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("nonlinear-beams")
class NonlinearBeamsPDE(ScalarPDEPreset):
    """Overdamped beam with state-dependent stiffness.

    Based on visualpde.com formulation (adapted to 2D):

        du/dt = -laplace(E(u) * laplace(u))

    where the stiffness E depends on local curvature:

        E(u) = E_star + Delta_E * (1 + tanh(laplace(u)/eps)) / 2

    For large curvatures (|laplace(u)| >> eps):
        - Positive curvature: E ≈ E_star + Delta_E (stiffer)
        - Negative curvature: E ≈ E_star (softer)

    This creates differential stiffness that leads to interesting dynamics
    where regions of high curvature behave differently than flat regions.

    Reference: https://visualpde.com/nonlinear-physics/nonlinear-beams
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="nonlinear-beams",
            category="physics",
            description="Overdamped beam with state-dependent stiffness",
            equations={
                "u": "-laplace(E(u) * laplace(u))",
                "E(u)": "E_star + Delta_E * (1 + tanh(laplace(u)/eps)) / 2",
            },
            parameters=[
                PDEParameter(
                    name="E_star",
                    default=1.0,
                    description="Baseline stiffness",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Delta_E",
                    default=5.0,
                    description="Stiffness variation (curvature-dependent)",
                    min_value=0.0,
                    max_value=24.0,
                ),
                PDEParameter(
                    name="eps",
                    default=1.0,
                    description="Curvature sensitivity",
                    min_value=0.1,
                    max_value=5.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="https://visualpde.com/nonlinear-physics/nonlinear-beams",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        E_star = parameters.get("E_star", 1.0)
        Delta_E = parameters.get("Delta_E", 5.0)
        eps = parameters.get("eps", 1.0)

        # Overdamped beam with state-dependent stiffness:
        # du/dt = -laplace(E(u) * laplace(u))
        # where E(u) = E_star + Delta_E * (1 + tanh(laplace(u)/eps)) / 2
        #
        # Simplified approximation (avoids nested laplacian of laplacian):
        # We approximate E as spatially uniform but curvature-dependent,
        # which gives: du/dt = -E(u) * laplace(laplace(u))
        #
        # For the full equation we'd need custom fields, so we use:
        # du/dt = -(E_star + Delta_E/2) * laplace(laplace(u))
        #         - (Delta_E/2) * tanh(laplace(u)/eps) * laplace(laplace(u))

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
    ) -> ScalarField:
        """Create initial beam displacement."""
        if ic_type in ("nonlinear-beams-default", "default"):
            # Create a smooth initial displacement
            np.random.seed(ic_params.get("seed"))
            amplitude = ic_params.get("amplitude", 0.5)

            x, y = np.meshgrid(
                np.linspace(0, 2 * np.pi, grid.shape[0]),
                np.linspace(0, 2 * np.pi, grid.shape[1]),
                indexing="ij",
            )

            # Sinusoidal initial shape with some noise
            data = amplitude * np.sin(x) * np.sin(y)
            data += 0.05 * np.random.randn(*grid.shape)

            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

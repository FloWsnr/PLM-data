"""Duffing oscillator (diffusively coupled)."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("duffing")
class DuffingPDE(MultiFieldPDEPreset):
    """Diffusively coupled forced Duffing oscillator.

    Based on visualpde.com formulation:

        dX/dt = epsilon * D * laplace(X) + Y
        dY/dt = D * laplace(X) - delta * Y - alpha * X - beta * X^3 + gamma * cos(omega * t)

    A spatially extended forced nonlinear oscillator exhibiting chaos,
    jump phenomena, and hysteresis through the interplay of cubic stiffness
    and periodic forcing.

    Key phenomena:
        - Hardening/softening frequency response: resonance peak bends
        - Jump phenomena and hysteresis: discontinuous amplitude changes
        - Period doubling to chaos: route through successive bifurcations
        - Strange attractors: complex fractal structure in phase space
        - Coexisting solutions: multiple stable states for same parameters

    Spring classification:
        - beta > 0, alpha > 0: Hardening spring
        - beta > 0, alpha < 0: Double-well potential (bistable)
        - beta < 0, alpha > 0: Softening spring

    Applications:
        - Mechanical oscillators with geometric nonlinearity
        - Magnetic pendulum between two magnets
        - MEMS nonlinear resonators
        - Electrical circuits with nonlinear inductors

    Reference: Duffing (1918), Guckenheimer & Holmes (1983)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="duffing",
            category="physics",
            description="Diffusively coupled forced Duffing oscillator",
            equations={
                "X": "epsilon * D * laplace(X) + Y",
                "Y": "D * laplace(X) - delta * Y - alpha * X - beta * X**3 + gamma * cos(omega * t)",
            },
            parameters=[
                PDEParameter("D", "Diffusion coupling strength"),
                PDEParameter("alpha", "Linear stiffness (negative = double-well)"),
                PDEParameter("beta", "Cubic nonlinearity strength"),
                PDEParameter("delta", "Linear damping coefficient"),
                PDEParameter("gamma", "Forcing amplitude"),
                PDEParameter("omega", "Forcing frequency"),
                PDEParameter("epsilon", "Artificial diffusion (numerical stability)"),
            ],
            num_fields=2,
            field_names=["X", "Y"],
            reference="Duffing (1918), Guckenheimer & Holmes (1983)",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Duffing PDE system.

        Note: The time-dependent forcing term requires explicit handling.
        For py-pde, we use the time variable 't' in the equation string.

        Args:
            parameters: Dictionary with D, alpha, beta, delta, gamma, omega, epsilon.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        D = parameters.get("D", 0.5)
        alpha = parameters.get("alpha", -1.0)
        beta = parameters.get("beta", 0.25)
        delta = parameters.get("delta", 0.1)
        gamma = parameters.get("gamma", 2.0)
        omega = parameters.get("omega", 2.0)
        epsilon = parameters.get("epsilon", 0.01)

        # Duffing oscillator with forcing
        # dX/dt = eps*D*laplace(X) + Y
        # dY/dt = D*laplace(X) - delta*Y - alpha*X - beta*X^3 + gamma*cos(omega*t)
        return PDE(
            rhs={
                "X": f"{epsilon} * {D} * laplace(X) + Y",
                "Y": f"{D} * laplace(X) - {delta} * Y - {alpha} * X - {beta} * X**3 + {gamma} * cos({omega} * t)",
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
        """Create initial oscillator states.

        Default: random perturbations (representing different starting positions).
        """
        seed = ic_params.get("seed")
        if seed is not None:
            np.random.seed(seed)
        noise = ic_params.get("noise", 1.0)

        # Random initial positions
        x_data = noise * np.random.randn(*grid.shape)
        # Start at rest
        y_data = np.zeros(grid.shape)

        X_field = ScalarField(grid, x_data)
        X_field.label = "X"
        Y_field = ScalarField(grid, y_data)
        Y_field.label = "Y"

        return FieldCollection([X_field, Y_field])

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        D = parameters.get("D", 0.5)
        alpha = parameters.get("alpha", -1.0)
        beta = parameters.get("beta", 0.25)
        delta = parameters.get("delta", 0.1)
        gamma = parameters.get("gamma", 2.0)
        omega = parameters.get("omega", 2.0)
        epsilon = parameters.get("epsilon", 0.01)

        return {
            "X": f"{epsilon} * {D} * laplace(X) + Y",
            "Y": f"{D} * laplace(X) - {delta} * Y - {alpha} * X - {beta} * X**3 + {gamma} * cos({omega} * t)",
        }

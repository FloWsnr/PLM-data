"""Van der Pol oscillator (diffusively coupled)."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("van-der-pol")
class VanDerPolPDE(MultiFieldPDEPreset):
    """Diffusively coupled Van der Pol oscillator.

    Based on visualpde.com formulation:

        dX/dt = Y
        dY/dt = D * (laplace(X) + epsilon * laplace(Y)) + mu * (1 - X^2) * Y - X

    A spatially extended version of the classic nonlinear oscillator,
    exhibiting self-sustained limit cycle oscillations coupled through
    diffusion to produce complex spatiotemporal patterns.

    Key features:
        - Self-sustained oscillations: energy pumped in at small amplitudes,
          dissipated at large amplitudes
        - Nonlinear damping: coefficient (1-X^2) changes sign with amplitude
        - Limit cycle: all trajectories converge to a single periodic orbit
        - Relaxation oscillations: for large mu, slow buildup then fast jumps

    Physical applications:
        - Electrical circuits (vacuum tube oscillators)
        - Heart rhythms (cardiac pacemaker models)
        - Neural oscillations
        - Coupled pendulums

    Historical note: Van der Pol and van der Mark (1927) observed "irregular noise"
    at certain drive frequencies - one of the first observations of chaos.

    Reference: Van der Pol (1926), FitzHugh (1961)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="van-der-pol",
            category="physics",
            description="Diffusively coupled Van der Pol oscillators",
            equations={
                "X": "Y",
                "Y": "D * (laplace(X) + epsilon * laplace(Y)) + mu * (1 - X**2) * Y - X",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Diffusion coupling strength",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="mu",
                    default=8.53,
                    description="Nonlinear damping parameter",
                    min_value=1.0,
                    max_value=30.0,
                ),
                PDEParameter(
                    name="epsilon",
                    default=0.005,
                    description="Artificial diffusion (numerical stability)",
                    min_value=0.0,
                    max_value=0.1,
                ),
            ],
            num_fields=2,
            field_names=["X", "Y"],
            reference="Van der Pol (1926) On relaxation oscillations",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Van der Pol PDE system.

        Args:
            parameters: Dictionary with D, mu, epsilon.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        D = parameters.get("D", 1.0)
        mu = parameters.get("mu", 8.53)
        epsilon = parameters.get("epsilon", 0.005)

        return PDE(
            rhs={
                "X": "Y",
                "Y": f"{D} * (laplace(X) + {epsilon} * laplace(Y)) + {mu} * (1 - X**2) * Y - X",
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

        Default: small random perturbations around zero.
        """
        seed = ic_params.get("seed")
        if seed is not None:
            np.random.seed(seed)
        noise = ic_params.get("noise", 0.05)

        x_data = noise * np.random.randn(*grid.shape)
        y_data = np.zeros(grid.shape)  # Start at rest

        X_field = ScalarField(grid, x_data)
        X_field.label = "X"
        Y_field = ScalarField(grid, y_data)
        Y_field.label = "Y"

        return FieldCollection([X_field, Y_field])

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        D = parameters.get("D", 1.0)
        mu = parameters.get("mu", 8.53)
        epsilon = parameters.get("epsilon", 0.005)

        return {
            "X": "Y",
            "Y": f"{D} * (laplace(X) + {epsilon} * laplace(Y)) + {mu} * (1 - X**2) * Y - X",
        }

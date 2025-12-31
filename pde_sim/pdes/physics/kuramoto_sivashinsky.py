"""Kuramoto-Sivashinsky equation for chaotic pattern formation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import PDEMetadata, PDEParameter, ScalarPDEPreset
from .. import register_pde


@register_pde("kuramoto-sivashinsky")
class KuramotoSivashinskyPDE(ScalarPDEPreset):
    """Kuramoto-Sivashinsky equation.

    A canonical model for spatiotemporal chaos:

        du/dt = -laplace(u) - nu * laplace(laplace(u)) - 0.5 * |grad(u)|^2

    or equivalently:

        du/dt = -laplace(u) - nu * laplace(laplace(u)) - 0.5 * gradient_squared(u)

    where:
        - u is the field variable (e.g., flame front position)
        - nu is the fourth-order diffusion coefficient

    This equation exhibits:
        - Linear instability at intermediate wavelengths
        - Nonlinear saturation
        - Spatiotemporal chaos

    Reference: Kuramoto (1978), Sivashinsky (1977)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="kuramoto-sivashinsky",
            category="physics",
            description="Kuramoto-Sivashinsky chaotic pattern equation",
            equations={"u": "-laplace(u) - nu * laplace(laplace(u)) - 0.5 * gradient_squared(u)"},
            parameters=[
                PDEParameter(
                    name="nu",
                    default=1.0,
                    description="Fourth-order diffusion coefficient",
                    min_value=0.1,
                    max_value=10.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Kuramoto-Sivashinsky spatiotemporal chaos",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Kuramoto-Sivashinsky equation PDE.

        Args:
            parameters: Dictionary containing 'nu' coefficient.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        nu = parameters.get("nu", 1.0)

        bc_spec = "periodic" if bc.get("x") == "periodic" else "no-flux"

        # py-pde uses gradient_squared for |grad(u)|^2
        return PDE(
            rhs={"u": f"-laplace(u) - {nu} * laplace(laplace(u)) - 0.5 * gradient_squared(u)"},
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial state - typically small random perturbations."""
        if ic_type in ("kuramoto-sivashinsky-default", "default"):
            # Small random perturbations around zero
            amplitude = ic_params.get("amplitude", 0.1)
            np.random.seed(ic_params.get("seed"))
            data = amplitude * np.random.randn(*grid.shape)
            return ScalarField(grid, data)

        # Fall back to parent implementation
        return create_initial_condition(grid, ic_type, ic_params)

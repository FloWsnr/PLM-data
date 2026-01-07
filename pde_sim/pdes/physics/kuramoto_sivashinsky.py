"""Kuramoto-Sivashinsky equation for spatiotemporal chaos."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import PDEMetadata, PDEParameter, ScalarPDEPreset
from .. import register_pde


@register_pde("kuramoto-sivashinsky")
class KuramotoSivashinskyPDE(ScalarPDEPreset):
    """Kuramoto-Sivashinsky equation for spatiotemporal chaos.

    Based on visualpde.com formulation:

        du/dt = -laplace(u) - laplace(laplace(u)) - |grad(u)|^2

    A fourth-order PDE exhibiting spatiotemporal chaos - one of the simplest
    equations known to produce turbulent-like dynamics.

    Physical contexts:
        - Flame fronts: wrinkling and cellular instabilities
        - Thin film flows: instability of liquid films
        - Chemical oscillations: phase dynamics
        - Plasma physics: edge turbulence

    Key features:
        - Negative diffusion (-laplace(u)): creates short-wavelength instability
        - Hyperdiffusion (-laplace(laplace(u))): provides large-wavenumber damping
        - Nonlinearity (-|grad(u)|^2): transfers energy between scales

    The balance produces chaos with characteristic wavelength - irregular but not random.

    Reference: Kuramoto (1978), Sivashinsky (1977)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="kuramoto-sivashinsky",
            category="physics",
            description="Kuramoto-Sivashinsky spatiotemporal chaos",
            equations={
                "u": "-laplace(u) - laplace(laplace(u)) - gradient_squared(u)",
            },
            parameters=[
                PDEParameter(
                    name="a",
                    default=0.03,
                    description="Damping coefficient (for numerical stability)",
                    min_value=0.0,
                    max_value=0.5,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Kuramoto (1978), Sivashinsky (1977)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Kuramoto-Sivashinsky equation PDE.

        Args:
            parameters: Dictionary containing 'a' damping coefficient.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        a = parameters.get("a", 0.03)

        # Kuramoto-Sivashinsky equation:
        # du/dt = -laplace(u) - laplace(laplace(u)) - |grad(u)|^2 - a*u
        # The small damping term -a*u helps with numerical stability
        rhs = "-laplace(u) - laplace(laplace(u)) - gradient_squared(u)"
        if a > 0:
            rhs += f" - {a} * u"

        return PDE(
            rhs={"u": rhs},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> ScalarField:
        """Create initial state - typically small random perturbations.

        Default: constant value with localized perturbations.
        """
        if ic_type in ("kuramoto-sivashinsky-default", "default"):
            # Small random perturbations around zero
            amplitude = ic_params.get("amplitude", 0.1)
            seed = ic_params.get("seed")
            if seed is not None:
                np.random.seed(seed)
            data = amplitude * np.random.randn(*grid.shape)
            return ScalarField(grid, data)

        # Fall back to parent implementation
        return create_initial_condition(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        a = parameters.get("a", 0.03)
        rhs = "-laplace(u) - laplace(laplace(u)) - gradient_squared(u)"
        if a > 0:
            rhs += f" - {a} * u"
        return {"u": rhs}

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

    Standard form:

        du/dt = -u * u_x - u_xx - u_xxxx

    A fourth-order PDE exhibiting spatiotemporal chaos - one of the simplest
    equations known to produce turbulent-like dynamics.

    Physical contexts:
        - Flame fronts: wrinkling and cellular instabilities
        - Thin film flows: instability of liquid films
        - Chemical oscillations: phase dynamics
        - Plasma physics: edge turbulence

    Key features:
        - Negative diffusion (-u_xx): creates short-wavelength instability
        - Hyperdiffusion (-u_xxxx): provides large-wavenumber damping
        - Nonlinearity (-u*u_x): conservative energy transfer between scales

    The balance produces chaos with characteristic wavelength - irregular but not random.
    The conservative nonlinearity preserves the spatial mean of u.

    Reference: Kuramoto (1978), Sivashinsky (1977)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="kuramoto-sivashinsky",
            category="physics",
            description="Kuramoto-Sivashinsky spatiotemporal chaos",
            equations={
                "u": "-0.5 * d_dx(u**2) - laplace(u) - laplace(laplace(u))",
            },
            parameters=[
                PDEParameter("a", "Damping coefficient (optional, 0 for standard KS)"),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Kuramoto (1978), Sivashinsky (1977)",
            supported_dimensions=[1, 2, 3],
        )

    def _build_rhs(self, a: float, ndim: int) -> str:
        """Build the RHS string for the KS equation.

        In 1D, uses conservative form -0.5 * d_dx(u^2) which preserves the mean.
        In 2D/3D, uses -0.5 * gradient_squared(u) = -0.5 * |grad(u)|^2.
        """
        if ndim == 1:
            rhs = "-0.5 * d_dx(u**2) - laplace(u) - laplace(laplace(u))"
        else:
            rhs = "-0.5 * gradient_squared(u) - laplace(u) - laplace(laplace(u))"
        if a > 0:
            rhs += f" - {a} * u"
        return rhs

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Kuramoto-Sivashinsky equation PDE.

        In 1D uses the conservative form -0.5 * d_dx(u^2) which preserves the mean.
        In multi-D uses -0.5 * |grad(u)|^2.

        Args:
            parameters: Dictionary containing 'a' damping coefficient.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        a = parameters["a"]
        rhs = self._build_rhs(a, grid.num_axes)

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
            rng = np.random.default_rng(ic_params.get("seed"))
            data = amplitude * rng.standard_normal(grid.shape)
            return ScalarField(grid, data)

        # Fall back to parent implementation
        return create_initial_condition(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        a = parameters["a"]
        rhs = "-0.5 * d_dx(u**2) - laplace(u) - laplace(laplace(u))"
        if a > 0:
            rhs += f" - {a} * u"
        return {"u": rhs}

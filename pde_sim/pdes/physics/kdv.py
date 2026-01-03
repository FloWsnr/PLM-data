"""Korteweg-de Vries (KdV) equation for solitons."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import PDEMetadata, PDEParameter, ScalarPDEPreset
from .. import register_pde


@register_pde("kdv")
class KdVPDE(ScalarPDEPreset):
    """Korteweg-de Vries (KdV) equation.

    The KdV equation describes shallow water waves and solitons:

        du/dt = -c * d_dx(u) - alpha * u * d_dx(u) - beta * d_dx(d_dx(d_dx(u)))

    or more commonly written as:

        du/dt = -6 * u * d_dx(u) - d_dx(d_dx(d_dx(u)))

    where:
        - u is the wave amplitude
        - c is the wave speed
        - alpha is the nonlinear coefficient
        - beta is the dispersion coefficient

    Supports soliton solutions.

    Note: This is a 1D equation extended to 2D. For pure 1D behavior,
    use initial conditions that vary only in x.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="kdv",
            category="physics",
            description="Korteweg-de Vries soliton equation",
            equations={"u": "-c * d_dx(u) - alpha * u * d_dx(u) - beta * d_dx(d_dx(d_dx(u)))"},
            parameters=[
                PDEParameter(
                    name="c",
                    default=0.0,
                    description="Linear wave speed",
                    min_value=-2.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="alpha",
                    default=6.0,
                    description="Nonlinear coefficient",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="beta",
                    default=1.0,
                    description="Dispersion coefficient",
                    min_value=0.1,
                    max_value=2.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Korteweg & de Vries (1895)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        c = parameters.get("c", 0.0)
        alpha = parameters.get("alpha", 6.0)
        beta = parameters.get("beta", 1.0)

        # Third derivative: d_dx(d_dx(d_dx(u)))
        # Linear term
        linear = f"-{c} * d_dx(u)" if c != 0 else ""
        # Nonlinear term
        nonlinear = f" - {alpha} * u * d_dx(u)"
        # Dispersion term
        dispersion = f" - {beta} * d_dx(d_dx(d_dx(u)))"

        rhs = (linear + nonlinear + dispersion).lstrip(" -")
        if rhs.startswith("- "):
            rhs = "-" + rhs[2:]

        return PDE(
            rhs={"u": rhs if rhs else "0"},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial state - soliton or wave packet."""
        if ic_type in ("kdv-default", "soliton"):
            # Soliton solution: u = A * sech^2(B*(x - x0))
            A = ic_params.get("amplitude", 1.0)
            width = ic_params.get("width", 0.1)
            x0 = ic_params.get("x0", 0.5)

            x, y = np.meshgrid(
                np.linspace(0, 1, grid.shape[0]),
                np.linspace(0, 1, grid.shape[1]),
                indexing="ij",
            )

            # sech^2 profile
            data = A / np.cosh((x - x0) / width) ** 2

            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

"""Complex Ginzburg-Landau equation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import PDEMetadata, PDEParameter, ScalarPDEPreset
from .. import register_pde


@register_pde("ginzburg-landau")
class GinzburgLandauPDE(ScalarPDEPreset):
    """Complex Ginzburg-Landau equation.

    A universal model for pattern formation near instability:

        dA/dt = A + (1 + i*c1) * laplace(A) - (1 + i*c3) * |A|^2 * A

    where:
        - A is the complex amplitude
        - c1 is the linear dispersion coefficient
        - c3 is the nonlinear dispersion coefficient

    For c1 = c3 = 0, reduces to the real Ginzburg-Landau equation.
    Exhibits spiral waves, defect chaos, and other patterns.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="ginzburg-landau",
            category="physics",
            description="Complex Ginzburg-Landau pattern formation",
            equations={"A": "A + (1 + 1j*c1) * laplace(A) - (1 + 1j*c3) * |A|^2 * A"},
            parameters=[
                PDEParameter(
                    name="c1",
                    default=0.0,
                    description="Linear dispersion coefficient",
                    min_value=-5.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="c3",
                    default=0.0,
                    description="Nonlinear dispersion coefficient",
                    min_value=-5.0,
                    max_value=5.0,
                ),
            ],
            num_fields=1,
            field_names=["A"],
            reference="Ginzburg-Landau pattern formation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        c1 = parameters.get("c1", 0.0)
        c3 = parameters.get("c3", 0.0)

        # Complex Ginzburg-Landau equation
        # Note: py-pde handles complex numbers with 1j notation
        return PDE(
            rhs={"A": f"A + (1 + 1j*{c1}) * laplace(A) - (1 + 1j*{c3}) * abs(A)**2 * A"},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial complex field with small perturbations."""
        if ic_type in ("ginzburg-landau-default", "default"):
            amplitude = ic_params.get("amplitude", 0.1)
            np.random.seed(ic_params.get("seed"))

            # Small complex perturbations
            real_part = amplitude * np.random.randn(*grid.shape)
            imag_part = amplitude * np.random.randn(*grid.shape)
            data = real_part + 1j * imag_part

            return ScalarField(grid, data, dtype=complex)

        field = create_initial_condition(grid, ic_type, ic_params)
        return ScalarField(grid, field.data.astype(complex), dtype=complex)

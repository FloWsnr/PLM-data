"""Porous Medium Equation for nonlinear degenerate diffusion."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("porous-medium")
class PorousMediumPDE(ScalarPDEPreset):
    """Porous Medium Equation for nonlinear degenerate diffusion.

    ∂u/∂t = Δ(u^m) where m > 1

    Models gas flow through porous media, groundwater infiltration,
    and population spreading with density-dependent diffusion.
    The equation is degenerate parabolic: diffusion vanishes where u=0,
    producing compactly supported solutions with sharp fronts (free boundaries).

    Reference: Vázquez (2007) "The Porous Medium Equation"
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="porous-medium",
            category="physics",
            description="Nonlinear degenerate diffusion with compactly supported solutions",
            equations={
                "u": "laplace(u**m)",
            },
            parameters=[
                PDEParameter("m", "Nonlinearity exponent (m > 1)"),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Vázquez (2007) The Porous Medium Equation",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        m = parameters["m"]

        rhs = f"laplace(u**{m})"

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
        """Create initial state for Porous Medium Equation.

        Default: Barenblatt-like compactly-supported parabolic bump.
        All ICs are clipped to non-negative values since u represents density.
        """
        if ic_type in ("porous-medium-default", "default"):
            amplitude = ic_params["amplitude"]
            radius = ic_params["radius"]
            rng = np.random.default_rng(ic_params.get("seed"))

            # Compute squared distance from domain center
            coords = grid.cell_coords
            center = np.array([(b[0] + b[1]) / 2 for b in grid.axes_bounds])
            r_sq = np.sum((coords - center) ** 2, axis=-1)

            # Compactly-supported parabolic bump: max(amplitude * (1 - r²/radius²), 0)
            data = amplitude * np.maximum(1.0 - r_sq / radius**2, 0.0)

            # Add tiny noise for dimension variation tests
            data += 1e-6 * rng.standard_normal(grid.shape)
            data = np.maximum(data, 0.0)

            return ScalarField(grid, data)

        # For generic ICs, clip to non-negative
        randomize = kwargs.get("randomize", False)
        field = create_initial_condition(grid, ic_type, ic_params, randomize=randomize)
        field.data = np.maximum(field.data, 0.0)
        return field

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        m = parameters["m"]
        return {
            "u": f"laplace(u**{m})",
        }

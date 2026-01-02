"""Superlattice pattern formation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("superlattice")
class SuperlatticePDE(ScalarPDEPreset):
    """Superlattice pattern formation.

    Swift-Hohenberg with hexagonal modulation for superlattice patterns:

        du/dt = epsilon * u - (1 + laplace)^2 * u + g2 * u^2 - u^3

    With additional modulation to promote superlattice structures.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="superlattice",
            category="physics",
            description="Superlattice pattern formation",
            equations={"u": "epsilon * u - (1 + laplace)^2 * u + g2 * u^2 - u^3"},
            parameters=[
                PDEParameter(
                    name="epsilon",
                    default=0.1,
                    description="Control parameter",
                    min_value=-0.5,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="g2",
                    default=0.5,
                    description="Quadratic nonlinearity",
                    min_value=0.0,
                    max_value=2.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Superlattice patterns in nonlinear optics",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        epsilon = parameters.get("epsilon", 0.1)
        g2 = parameters.get("g2", 0.5)

        # (1 + laplace)^2 = 1 + 2*laplace + laplace(laplace)
        return PDE(
            rhs={
                "u": f"{epsilon} * u - u - 2 * laplace(u) - laplace(laplace(u)) + {g2} * u**2 - u**3"
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial hexagonal seed pattern."""
        if ic_type in ("superlattice-default", "default"):
            np.random.seed(ic_params.get("seed"))
            amplitude = ic_params.get("amplitude", 0.1)

            x, y = np.meshgrid(
                np.linspace(0, 2 * np.pi, grid.shape[0]),
                np.linspace(0, 2 * np.pi, grid.shape[1]),
                indexing="ij",
            )

            # Hexagonal seed + noise
            k = 1.0
            hex_pattern = (
                np.cos(k * x)
                + np.cos(k * (x / 2 + y * np.sqrt(3) / 2))
                + np.cos(k * (x / 2 - y * np.sqrt(3) / 2))
            )
            data = amplitude * (hex_pattern / 3 + 0.1 * np.random.randn(*grid.shape))
            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

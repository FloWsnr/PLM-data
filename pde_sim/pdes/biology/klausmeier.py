"""Klausmeier model for dryland vegetation patterns."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("klausmeier")
class KlausmeierPDE(MultiFieldPDEPreset):
    """Klausmeier vegetation-water model for dryland patterns.

    Describes banded vegetation patterns in semi-arid regions:

        dw/dt = a - w - w*n^2 + V*dw/dx + Dw*laplace(w)
        dn/dt = w*n^2 - m*n + Dn*laplace(n)

    where:
        - w is the water density
        - n is the plant biomass density
        - a is the rainfall rate
        - m is the plant mortality rate
        - V is the water advection velocity (downslope flow)
        - Dw, Dn are diffusion coefficients

    Key phenomena:
        - Stripe patterns (tiger bush) perpendicular to slope
        - Traveling waves migrating uphill
        - Spots, gaps, labyrinths depending on conditions

    References:
        Klausmeier, C. A. (1999). Science, 284(5421), 1826-1828.
        Rietkerk et al. (2002). American Naturalist, 160(4), 524-530.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="klausmeier",
            category="biology",
            description="Klausmeier vegetation-water model for dryland patterns",
            equations={
                "n": "Dn * laplace(n) + w * n**2 - m * n",
                "w": "Dw * laplace(w) + a - w - w * n**2 + V * d_dx(w)",
            },
            parameters=[
                PDEParameter(
                    name="a",
                    default=2.0,
                    description="Rainfall rate",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="m",
                    default=0.54,
                    description="Plant mortality rate",
                    min_value=0.2,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="V",
                    default=50.0,
                    description="Water advection velocity",
                    min_value=0.0,
                    max_value=200.0,
                ),
                PDEParameter(
                    name="Dn",
                    default=1.0,
                    description="Plant diffusion coefficient",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dw",
                    default=1.0,
                    description="Water diffusion coefficient",
                    min_value=0.1,
                    max_value=10.0,
                ),
            ],
            num_fields=2,
            field_names=["n", "w"],
            reference="Klausmeier (1999)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        a = parameters.get("a", 2.0)
        m = parameters.get("m", 0.54)
        V = parameters.get("V", 50.0)
        Dn = parameters.get("Dn", 1.0)
        Dw = parameters.get("Dw", 1.0)

        return PDE(
            rhs={
                "n": f"{Dn} * laplace(n) + w * n**2 - {m} * n",
                "w": f"{Dw} * laplace(w) + {a} - w - w * n**2 + {V} * d_dx(w)",
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
        """Create initial state for Klausmeier model."""
        np.random.seed(ic_params.get("seed"))

        # Initial plants: random uniform [0,1] (matching Visual PDE "RAND")
        n_data = np.random.rand(*grid.shape)

        # Initial water: constant 1 (matching Visual PDE reference)
        w_data = np.ones(grid.shape)

        n = ScalarField(grid, n_data)
        n.label = "n"
        w = ScalarField(grid, w_data)
        w.label = "w"

        return FieldCollection([n, w])

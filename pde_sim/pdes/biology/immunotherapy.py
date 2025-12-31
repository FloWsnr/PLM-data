"""Tumor-immune interaction model."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("immunotherapy")
class ImmunotherapyPDE(MultiFieldPDEPreset):
    """Tumor-immune interaction model.

    Simplified tumor-immune dynamics with therapy:

        dT/dt = Dt * laplace(T) + rT * T * (1 - T/K) - k * T * I
        dI/dt = Di * laplace(I) + s + rI * T * I / (a + T) - dI * I

    where:
        - T is tumor cell density
        - I is immune cell density
        - rT is tumor growth rate
        - k is kill rate by immune cells
        - s is immune cell source (therapy)
        - rI is immune recruitment rate
        - dI is immune cell death rate
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="immunotherapy",
            category="biology",
            description="Tumor-immune interaction dynamics",
            equations={
                "T": "Dt * laplace(T) + rT * T * (1 - T/K) - k * T * I",
                "I": "Di * laplace(I) + s + rI * T * I / (a + T) - dI * I",
            },
            parameters=[
                PDEParameter(
                    name="Dt",
                    default=0.01,
                    description="Tumor diffusion",
                    min_value=0.001,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="Di",
                    default=0.1,
                    description="Immune cell diffusion",
                    min_value=0.001,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="rT",
                    default=1.0,
                    description="Tumor growth rate",
                    min_value=0.1,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="K",
                    default=1.0,
                    description="Tumor carrying capacity",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="k",
                    default=1.0,
                    description="Immune kill rate",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="s",
                    default=0.1,
                    description="Immune source (therapy)",
                    min_value=0.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="rI",
                    default=0.5,
                    description="Immune recruitment rate",
                    min_value=0.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="a",
                    default=0.1,
                    description="Half-saturation constant",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="dI",
                    default=0.3,
                    description="Immune death rate",
                    min_value=0.01,
                    max_value=2.0,
                ),
            ],
            num_fields=2,
            field_names=["T", "I"],
            reference="Tumor-immune system dynamics",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Dt = parameters.get("Dt", 0.01)
        Di = parameters.get("Di", 0.1)
        rT = parameters.get("rT", 1.0)
        K = parameters.get("K", 1.0)
        k = parameters.get("k", 1.0)
        s = parameters.get("s", 0.1)
        rI = parameters.get("rI", 0.5)
        a = parameters.get("a", 0.1)
        dI = parameters.get("dI", 0.3)

        T_rhs = f"{Dt} * laplace(T) + {rT} * T * (1 - T / {K}) - {k} * T * I"
        I_rhs = f"{Di} * laplace(I) + {s} + {rI} * T * I / ({a} + T) - {dI} * I"

        return PDE(
            rhs={"T": T_rhs, "I": I_rhs},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial tumor with some immune cells."""
        np.random.seed(ic_params.get("seed"))

        # Small tumor in center
        x, y = np.meshgrid(
            np.linspace(0, 1, grid.shape[0]),
            np.linspace(0, 1, grid.shape[1]),
            indexing="ij",
        )
        r_sq = (x - 0.5) ** 2 + (y - 0.5) ** 2
        T_data = 0.5 * np.exp(-r_sq / 0.02)

        # Uniform low immune presence
        I_data = 0.1 * np.ones(grid.shape)

        T = ScalarField(grid, T_data)
        T.label = "T"
        I = ScalarField(grid, I_data)
        I.label = "I"

        return FieldCollection([T, I])

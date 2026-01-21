"""Harsh environment model for population survival thresholds."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("harsh-environment")
class HarshEnvironmentPDE(ScalarPDEPreset):
    """Logistic growth with hostile boundaries.

    Explores how boundary conditions affect population persistence:

        du/dt = D*laplace(u) + r*u*(1 - u/K)

    With Dirichlet (hostile) boundaries, there is a critical diffusion:
        Dc = L^2 / (2*pi^2)

    For a domain of L = 10: Dc ~ 5.07

    Key behaviors:
        - D < Dc: Population persists (positive equilibrium)
        - D > Dc: Population goes extinct
        - Near Dc: Very long transients

    The carrying capacity K affects equilibrium density but NOT
    the persistence threshold.

    References:
        Skellam (1951). Biometrika, 38(1/2), 196-218.
        Kierstead & Slobodkin (1953). J. Marine Research, 12(1), 141-147.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="harsh-environment",
            category="biology",
            description="Logistic growth with harsh boundary conditions",
            equations={"u": "D * laplace(u) + r * u * (1 - u/K)"},
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.01,
                    description="Diffusion coefficient",
                    min_value=0.001,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="r",
                    default=1.0,
                    description="Growth rate",
                    min_value=0.1,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="K",
                    default=1.0,
                    description="Carrying capacity",
                    min_value=0.1,
                    max_value=1000.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Skellam (1951), Kierstead & Slobodkin (1953)",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 0.01)
        r = parameters.get("r", 1.0)
        K = parameters.get("K", 1.0)

        return PDE(
            rhs={"u": f"{D} * laplace(u) + {r} * u * (1 - u / {K})"},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> ScalarField:
        """Create initial population - sparse random nucleation sites.

        Reference: Visual PDE uses 0.1*exp(-10000*RAND) which creates
        sparse nucleation sites where random values happen to be near 0.
        """
        if ic_type in ("harsh-environment-default", "default"):
            # Sparse random nucleation sites (matches Visual PDE reference)
            np.random.seed(ic_params.get("seed"))
            rand_field = np.random.random(grid.shape)
            # 0.1 * exp(-10000 * RAND) creates sparse nucleation where RAND ~ 0
            data = 0.1 * np.exp(-10000 * rand_field)
            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

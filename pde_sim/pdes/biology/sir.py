"""SIR epidemiological model with spatial diffusion."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("sir")
class SIRModelPDE(MultiFieldPDEPreset):
    """SIR epidemiological model with spatial diffusion.

    The spatially coupled SIR model describes disease spread dynamics:

        ds/dt = D * laplace(s) - beta * i * s
        di/dt = D * laplace(i) + beta * i * s - gamma * i
        dr/dt = D * laplace(r) + gamma * i

    where:
        - s is the density of susceptible individuals
        - i is the density of infected individuals
        - r is the density of recovered individuals
        - D is the diffusivity (spatial mobility)
        - beta is the transmission rate
        - gamma is the recovery rate

    Conservation: s + i + r = constant (total population conserved)

    The model exhibits:
        - Traveling epidemic waves
        - Endemic steady states
        - Herd immunity thresholds
        - R0 = beta/gamma (basic reproduction number)

    Reference: Kermack-McKendrick (1927) with spatial diffusion
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="sir",
            category="biology",
            description="SIR epidemiological model with spatial diffusion",
            equations={
                "s": "D * laplace(s) - beta * i * s",
                "i": "D * laplace(i) + beta * i * s - gamma * i",
                "r": "D * laplace(r) + gamma * i",
            },
            parameters=[
                PDEParameter(
                    name="beta",
                    default=2.0,
                    description="Transmission rate",
                    min_value=0.1,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="gamma",
                    default=0.1,
                    description="Recovery rate",
                    min_value=0.01,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="D",
                    default=0.05,
                    description="Diffusivity (spatial mobility)",
                    min_value=0.001,
                    max_value=0.3,
                ),
            ],
            num_fields=3,
            field_names=["s", "i", "r"],
            reference="Kermack-McKendrick epidemic model",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the SIR epidemic PDE system.

        Args:
            parameters: Dictionary containing 'beta', 'gamma', 'D'.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        beta = parameters.get("beta", 2.0)
        gamma = parameters.get("gamma", 0.1)
        D = parameters.get("D", 0.05)

        return PDE(
            rhs={
                "s": f"{D} * laplace(s) - {beta} * i * s",
                "i": f"{D} * laplace(i) + {beta} * i * s - {gamma} * i",
                "r": f"{D} * laplace(r) + {gamma} * i",
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state for SIR model.

        Common initial conditions:
        - default: Mostly susceptible with localized infection
        - random: Random initial infected regions
        """
        seed = ic_params.get("seed")
        if seed is not None:
            np.random.seed(seed)

        if ic_type in ("sir-default", "default", "localized"):
            # Most population susceptible, small infected region
            s_data = np.ones(grid.shape)
            i_data = np.zeros(grid.shape)
            r_data = np.zeros(grid.shape)

            # Localized infection at center or corner
            infection_location = ic_params.get("location", "corner")
            infection_size = ic_params.get("size", 3)

            if infection_location == "center":
                cx, cy = grid.shape[0] // 2, grid.shape[1] // 2
                i_data[
                    cx - infection_size : cx + infection_size,
                    cy - infection_size : cy + infection_size,
                ] = 1.0
                s_data[
                    cx - infection_size : cx + infection_size,
                    cy - infection_size : cy + infection_size,
                ] = 0.0
            else:  # corner
                i_data[0:infection_size, 0:infection_size] = 1.0
                s_data[0:infection_size, 0:infection_size] = 0.0

        elif ic_type == "random":
            # Random infected spots
            infection_fraction = ic_params.get("infection_fraction", 0.05)
            s_data = np.ones(grid.shape)
            i_data = np.zeros(grid.shape)
            r_data = np.zeros(grid.shape)

            # Random infection mask
            mask = np.random.random(grid.shape) < infection_fraction
            i_data[mask] = 1.0
            s_data[mask] = 0.0

        else:
            # Default fallback
            s_data = np.ones(grid.shape) * 0.95
            i_data = np.ones(grid.shape) * 0.05
            r_data = np.zeros(grid.shape)

        # Ensure non-negative and bounded
        s_data = np.clip(s_data, 0, 1)
        i_data = np.clip(i_data, 0, 1)
        r_data = np.clip(r_data, 0, 1)

        s = ScalarField(grid, s_data)
        s.label = "s"
        i = ScalarField(grid, i_data)
        i.label = "i"
        r = ScalarField(grid, r_data)
        r.label = "r"

        return FieldCollection([s, i, r])

"""Cahn-Hilliard equation for phase separation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("cahn-hilliard")
class CahnHilliardPDE(ScalarPDEPreset):
    """Cahn-Hilliard equation for phase separation with reaction term.

    Based on visualpde.com formulation:

        du/dt = r * laplace(u^3 - u - a * laplace(u)) + u - u^3

    This is the Cahn-Hilliard equation with an additional reaction term (u - u^3).
    The standard Cahn-Hilliard without reaction would be:

        du/dt = laplace(F'(u) - g * laplace(u))

    where F(u) = (1/4)(u^2-1)^2 is a double-well potential.

    Key phenomena:
        - Spinodal decomposition: spontaneous phase separation
        - Coarsening dynamics: larger domains grow at expense of smaller ones
        - Conservation of mass: average concentration is conserved
        - Interfacial energy: higher-order derivative controls interface width

    Parameters:
        - r: timescale parameter (mobility) - controls separation rate
        - a: interfacial energy coefficient (controls interface width)
        - D: diffusion coefficient

    Reference: Cahn & Hilliard (1958), Bray (2002)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="cahn-hilliard",
            category="physics",
            description="Cahn-Hilliard phase separation with reaction",
            equations={
                "u": "r * laplace(u**3 - u - a * laplace(u)) + u - u**3",
            },
            parameters=[
                PDEParameter(
                    name="r",
                    default=0.01,
                    description="Timescale parameter (mobility)",
                    min_value=0.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="a",
                    default=1.0,
                    description="Interfacial energy coefficient",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Diffusion coefficient",
                    min_value=0.1,
                    max_value=10.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Cahn & Hilliard (1958) Free Energy of a Nonuniform System",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Cahn-Hilliard PDE.

        Args:
            parameters: Dictionary with r, a, D.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        r = parameters.get("r", 0.01)
        a = parameters.get("a", 1.0)
        D = parameters.get("D", 1.0)

        # Cahn-Hilliard with reaction:
        # du/dt = r * laplace(u^3 - u - a * laplace(u)) + u - u^3
        # Scaling diffusion by D
        return PDE(
            rhs={"u": f"{r} * {D} * laplace(u**3 - u - {a} * laplace(u)) + u - u**3"},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> ScalarField:
        """Create initial state for Cahn-Hilliard.

        Default: random values near +/-1 (tanh(30*(RAND-0.5)))
        """
        if ic_type in ("cahn-hilliard-default", "default"):
            seed = ic_params.get("seed")
            if seed is not None:
                np.random.seed(seed)

            # Random values near +/-1 using tanh transform
            # This creates initial conditions near the two pure phases
            rand_data = np.random.rand(*grid.shape)
            data = np.tanh(30 * (rand_data - 0.5))

            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        r = parameters.get("r", 0.01)
        a = parameters.get("a", 1.0)
        D = parameters.get("D", 1.0)

        return {
            "u": f"{r} * {D} * laplace(u**3 - u - {a} * laplace(u)) + u - u**3",
        }

"""Bacteria advection model for transport in flowing water."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("bacteria-advection")
class BacteriaAdvectionPDE(ScalarPDEPreset):
    """Bacterial transport in a river reach.

    Simple advection-reaction model for bacterial concentration:

        dC/dt = -u * dC/dx - k*C

    where:
        - C is the bacterial concentration
        - u is the flow speed (advection velocity)
        - k is the decay rate (die-off coefficient)

    This is a first-order linear PDE with:
        - Advection term: transport downstream
        - Reaction term: exponential decay

    Steady-state solution: C(x) = c0 * exp(-k*x/u)

    Decay length scale: L = u/k

    References:
        Chapra (1997). Surface Water-Quality Modeling. McGraw-Hill.
        Bowie et al. (1985). EPA/600/3-85/040.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="bacteria-advection",
            category="biology",
            description="Bacterial transport with advection and decay",
            equations={"C": "-u * d_dx(C) - k * C"},
            parameters=[
                PDEParameter(
                    name="u",
                    default=0.62,
                    description="Flow speed (advection velocity)",
                    min_value=0.1,
                    max_value=4.0,
                ),
                PDEParameter(
                    name="k",
                    default=0.006,
                    description="Decay rate (die-off coefficient)",
                    min_value=0.0,
                    max_value=0.1,
                ),
                PDEParameter(
                    name="c0",
                    default=0.77,
                    description="Inlet concentration",
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            num_fields=1,
            field_names=["C"],
            reference="Chapra (1997)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        u = parameters.get("u", 0.62)
        k = parameters.get("k", 0.006)

        # Pure advection-decay (no diffusion)
        return PDE(
            rhs={"C": f"-{u} * d_dx(C) - {k} * C"},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> ScalarField:
        """Create initial bacterial concentration.

        Default IC is zero everywhere - the inlet boundary condition
        (Dirichlet = c0 at left) fills in the concentration over time.
        """
        c0 = ic_params.get("c0", 0.77)

        if ic_type in ("bacteria-advection-default", "default", "zero"):
            # Start with zero concentration (matches Visual PDE reference)
            # The left boundary condition provides inlet concentration c0
            return ScalarField(grid, np.zeros(grid.shape))

        if ic_type == "steady-state":
            # Analytical steady-state profile: C(x) = c0 * exp(-k*x/u)
            u = ic_params.get("u", 0.62)
            k = ic_params.get("k", 0.006)

            x_bounds = grid.axes_bounds[0]
            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])

            # For 2D, replicate across y
            if len(grid.shape) > 1:
                x = x[:, np.newaxis] * np.ones(grid.shape[1])

            data = c0 * np.exp(-k * x / max(u, 0.01))
            return ScalarField(grid, data)

        if ic_type == "uniform":
            return ScalarField(grid, c0 * np.ones(grid.shape))

        return create_initial_condition(grid, ic_type, ic_params)

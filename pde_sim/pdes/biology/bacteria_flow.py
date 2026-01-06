"""Bacteria in flowing river - advection-decay model."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("bacteria-flow")
class BacteriaFlowPDE(ScalarPDEPreset):
    """Bacteria in flowing river - advection-decay model.

    Simple model from visualpde.com for bacteria concentration in a river:

        dC/dt = -u*∂C/∂x - k*C

    where:
        - C is bacteria concentration
        - u is flow speed (downstream)
        - k is decay rate (e.g., from UV exposure)

    The inlet (left boundary) has a specified bacteria concentration c0.
    Bacteria are carried downstream and decay over time.

    This is a 1D advection-reaction equation - no diffusion term.
    The "harsh" aspect is the hostile environment causing decay.

    Reference: visualpde.com bacteria in flow
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="bacteria-flow",
            category="biology",
            description="Bacteria advection-decay in flowing river",
            equations={"C": "-u * d_dx(C) - k * C"},
            parameters=[
                PDEParameter(
                    name="u",
                    default=0.6,
                    description="Flow speed (downstream)",
                    min_value=0.1,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="k",
                    default=0.006,
                    description="Decay rate (UV, etc.)",
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
            reference="visualpde.com bacteria in flow",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        u = parameters.get("u", 0.6)
        k = parameters.get("k", 0.006)

        # Pure advection-decay: no diffusion
        rhs = f"-{u} * d_dx(C) - {k} * C"

        return PDE(
            rhs={"C": rhs},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial bacteria concentration."""
        c0 = ic_params.get("c0", 0.77)

        if ic_type in ("bacteria-flow-default", "default"):
            # Start with inlet concentration on left, zero elsewhere
            x, y = np.meshgrid(
                np.linspace(0, 1, grid.shape[0]),
                np.linspace(0, 1, grid.shape[1]),
                indexing="ij",
            )
            # Smooth inlet region
            data = c0 * np.exp(-((x - 0.1) ** 2) / 0.01)
            return ScalarField(grid, np.clip(data, 0, 1))

        return create_initial_condition(grid, ic_type, ic_params)

"""Standing wave equation - undamped wave for membrane vibration modes."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("wave-standing")
class StandingWavePDE(MultiFieldPDEPreset):
    """Standing wave equation for membrane vibration modes.

    This is the pure wave equation without any damping (C=0), which produces
    standing wave patterns when initialized with sinusoidal modes:

        du/dt = v
        dv/dt = D * laplace(u)

    where:
        - u is displacement
        - v is velocity (du/dt)
        - D is the wave speed squared (c^2)

    With fixed (Dirichlet) or reflecting (Neumann) boundary conditions and
    sinusoidal initial conditions, this produces Chladni-like patterns.

    Reference:
        https://visualpde.com/basic-pdes/wave-equation
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="wave-standing",
            category="basic",
            description="Standing wave equation for membrane vibration modes (Chladni figures)",
            equations={
                "u": "v",
                "v": "D * laplace(u)",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Wave speed squared (c^2)",
                    min_value=0.1,
                    max_value=100.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="https://visualpde.com/basic-pdes/wave-equation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)

        bc_spec = self._convert_bc(bc)

        return PDE(
            rhs={
                "u": "v",
                "v": f"{D} * laplace(u)",
            },
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial state for standing wave equation."""
        # u gets the specified initial condition
        u = create_initial_condition(grid, ic_type, ic_params)
        u.label = "u"

        # v (velocity) starts at zero by default
        v_data = np.zeros(grid.shape)
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

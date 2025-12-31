"""Nonlinear beam equation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("nonlinear-beams")
class NonlinearBeamsPDE(MultiFieldPDEPreset):
    """Nonlinear beam equation.

    Fourth-order wave equation with geometric nonlinearity:

        d²u/dt² = -D * laplace(laplace(u)) + alpha * laplace((du/dx)²)

    Converted to first-order system:
        du/dt = v
        dv/dt = -D * laplace(laplace(u)) + alpha * laplace(gradient_squared(u))
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="nonlinear-beams",
            category="physics",
            description="Nonlinear beam/plate vibrations",
            equations={
                "u": "v",
                "v": "-D * laplace(laplace(u)) + alpha * laplace(gradient_squared(u))",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Bending stiffness",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="alpha",
                    default=0.1,
                    description="Nonlinearity coefficient",
                    min_value=0.0,
                    max_value=2.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Nonlinear plate vibrations",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        alpha = parameters.get("alpha", 0.1)

        return PDE(
            rhs={
                "u": "v",
                "v": f"-{D} * laplace(laplace(u)) + {alpha} * laplace(gradient_squared(u))",
            },
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial beam displacement."""
        u = create_initial_condition(grid, ic_type, ic_params)
        u.label = "u"

        v_data = np.zeros(grid.shape)
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

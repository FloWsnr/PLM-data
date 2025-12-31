"""Bacteria with chemotaxis in flowing medium."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("bacteria-flow")
class BacteriaFlowPDE(MultiFieldPDEPreset):
    """Bacteria with chemotaxis in flowing medium.

    Bacteria moving up chemical gradients with advection:

        db/dt = Db * laplace(b) - chi * div(b * grad(c)) - v * d_dx(b) + r * b
        dc/dt = Dc * laplace(c) - k * c + s * b

    where:
        - b is bacteria density
        - c is chemoattractant concentration
        - chi is chemotaxis strength
        - v is flow velocity
        - r is bacteria growth rate
        - k is chemical decay rate
        - s is chemical production by bacteria
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="bacteria-flow",
            category="biology",
            description="Bacteria chemotaxis with flow",
            equations={
                "b": "Db * laplace(b) - chi * div(b * grad(c)) - v * d_dx(b) + r * b",
                "c": "Dc * laplace(c) - k * c + s * b",
            },
            parameters=[
                PDEParameter(
                    name="Db",
                    default=0.1,
                    description="Bacteria diffusion",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="Dc",
                    default=1.0,
                    description="Chemical diffusion",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="chi",
                    default=1.0,
                    description="Chemotaxis strength",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="v",
                    default=0.5,
                    description="Flow velocity",
                    min_value=0.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="r",
                    default=0.1,
                    description="Bacteria growth rate",
                    min_value=0.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="k",
                    default=0.1,
                    description="Chemical decay rate",
                    min_value=0.01,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="s",
                    default=0.1,
                    description="Chemical production rate",
                    min_value=0.0,
                    max_value=2.0,
                ),
            ],
            num_fields=2,
            field_names=["b", "c"],
            reference="Chemotaxis with advection",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Db = parameters.get("Db", 0.1)
        Dc = parameters.get("Dc", 1.0)
        chi = parameters.get("chi", 1.0)
        v = parameters.get("v", 0.5)
        r = parameters.get("r", 0.1)
        k = parameters.get("k", 0.1)
        s = parameters.get("s", 0.1)

        # Simplified chemotaxis (linearized around uniform c)
        b_rhs = f"{Db} * laplace(b) - {chi} * (d_dx(b) * d_dx(c) + d_dy(b) * d_dy(c))"
        if v != 0:
            b_rhs += f" - {v} * d_dx(b)"
        if r != 0:
            b_rhs += f" + {r} * b"

        c_rhs = f"{Dc} * laplace(c) - {k} * c + {s} * b"

        return PDE(
            rhs={"b": b_rhs, "c": c_rhs},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial bacteria colony with chemical."""
        np.random.seed(ic_params.get("seed"))

        x, y = np.meshgrid(
            np.linspace(0, 1, grid.shape[0]),
            np.linspace(0, 1, grid.shape[1]),
            indexing="ij",
        )

        # Bacteria colony
        r_sq = (x - 0.3) ** 2 + (y - 0.5) ** 2
        b_data = np.exp(-r_sq / 0.02)

        # Low uniform chemical
        c_data = 0.1 * np.ones(grid.shape)

        b = ScalarField(grid, b_data)
        b.label = "b"
        c = ScalarField(grid, c_data)
        c.label = "c"

        return FieldCollection([b, c])

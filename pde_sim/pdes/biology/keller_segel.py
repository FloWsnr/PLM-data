"""Keller-Segel chemotaxis model."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("keller-segel")
class KellerSegelPDE(MultiFieldPDEPreset):
    """Keller-Segel chemotaxis model.

    Describes cell movement in response to chemical signals:

        du/dt = Du * laplace(u) - chi * div(u * grad(c))
        dc/dt = Dc * laplace(c) + alpha * u - beta * c

    where:
        - u is the cell density
        - c is the chemoattractant concentration
        - Du, Dc are diffusion coefficients
        - chi is the chemotactic sensitivity
        - alpha is the chemoattractant production rate
        - beta is the chemoattractant decay rate

    This model can exhibit blow-up (aggregation) in finite time.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="keller-segel",
            category="biology",
            description="Keller-Segel chemotaxis model",
            equations={
                "u": "Du * laplace(u) - chi * div(u * gradient(c))",
                "c": "Dc * laplace(c) + alpha * u - beta * c",
            },
            parameters=[
                PDEParameter(
                    name="Du",
                    default=1.0,
                    description="Cell diffusion coefficient",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dc",
                    default=1.0,
                    description="Chemoattractant diffusion coefficient",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="chi",
                    default=1.0,
                    description="Chemotactic sensitivity",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="alpha",
                    default=1.0,
                    description="Chemoattractant production rate",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="beta",
                    default=1.0,
                    description="Chemoattractant decay rate",
                    min_value=0.01,
                    max_value=10.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "c"],
            reference="Keller & Segel (1970) chemotaxis",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Du = parameters.get("Du", 1.0)
        Dc = parameters.get("Dc", 1.0)
        chi = parameters.get("chi", 1.0)
        alpha = parameters.get("alpha", 1.0)
        beta = parameters.get("beta", 1.0)

        # Note: py-pde uses "gradient" not "grad"
        # The divergence of u*grad(c) requires careful handling
        # Using the approximation: div(u*grad(c)) ~ u*laplace(c) + grad(u).grad(c)
        u_rhs = f"{Du} * laplace(u) - {chi} * (u * laplace(c) + inner(gradient(u), gradient(c)))"
        c_rhs = f"{Dc} * laplace(c) + {alpha} * u - {beta} * c"

        return PDE(
            rhs={
                "u": u_rhs,
                "c": c_rhs,
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state with localized cell density."""
        # Default: use gaussian blobs for cells, uniform for chemoattractant
        if ic_type in ("keller-segel-default", "default"):
            np.random.seed(ic_params.get("seed"))

            # Cells - gaussian blobs
            u = create_initial_condition(
                grid, "gaussian-blobs", {"num_blobs": 3, "amplitude": 1.0, "width": 0.1}
            )
            u.label = "u"

            # Chemoattractant - start uniform
            c = ScalarField(grid, np.ones(grid.shape) * 0.1)
            c.label = "c"

            return FieldCollection([u, c])

        # Use parent implementation for other IC types
        return super().create_initial_state(grid, ic_type, ic_params)

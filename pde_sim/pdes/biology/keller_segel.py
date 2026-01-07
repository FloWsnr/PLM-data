"""Keller-Segel chemotaxis model for cell aggregation."""

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

        du/dt = laplace(u) - div(chi(u) * gradient(v)) + u*(1-u)
        dv/dt = D * laplace(v) + u - a*v

    where chi(u) = c*u/(1+u^2) is the chemotactic sensitivity.

    Components:
        - u is the cell density
        - v is the chemoattractant concentration
        - c is the chemotaxis strength
        - D is the chemical diffusion coefficient
        - a is the chemical decay rate

    Key features:
        - Saturating chemotaxis prevents blow-up
        - Logistic growth stabilizes cell density
        - Pattern formation for: 2*sqrt(aD) < c/2 - D - a

    References:
        Keller & Segel (1970). J. Theor. Biol., 26(3), 399-415.
        Keller & Segel (1971). J. Theor. Biol., 30(2), 225-234.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="keller-segel",
            category="biology",
            description="Keller-Segel chemotaxis with logistic growth",
            equations={
                "u": "laplace(u) - div(chi(u) * gradient(v)) + u * (1 - u)",
                "v": "D * laplace(v) + u - a * v",
            },
            parameters=[
                PDEParameter(
                    name="c",
                    default=4.0,
                    description="Chemotaxis strength",
                    min_value=3.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Chemical diffusion coefficient",
                    min_value=0.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="a",
                    default=0.1,
                    description="Chemical decay rate",
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Keller & Segel (1970)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        c = parameters.get("c", 4.0)
        D = parameters.get("D", 1.0)
        a = parameters.get("a", 0.1)

        # chi(u) = c*u/(1+u^2)
        # div(chi(u)*grad(v)) = chi(u)*laplace(v) + chi'(u)*grad(u)dot(grad(v))
        # chi'(u) = c*(1-u^2)/(1+u^2)^2
        chi_expr = f"{c} * u / (1 + u**2)"
        chi_deriv = f"{c} * (1 - u**2) / (1 + u**2)**2"

        u_rhs = (
            f"laplace(u) - ({chi_expr}) * laplace(v) "
            f"- ({chi_deriv}) * inner(gradient(u), gradient(v)) "
            f"+ u * (1 - u)"
        )
        v_rhs = f"{D} * laplace(v) + u - {a} * v"

        return PDE(
            rhs={
                "u": u_rhs,
                "v": v_rhs,
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
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
            v = ScalarField(grid, np.ones(grid.shape) * 0.1)
            v.label = "v"

            return FieldCollection([u, v])

        # Use parent implementation for other IC types
        return super().create_initial_state(grid, ic_type, ic_params)

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

    Standard formulation from visualpde.com with logistic growth:

        du/dt = ∇²u - ∇·(χ(u)∇v) + u(1-u)
        dv/dt = D∇²v + u - a*v

    where χ(u) = c*u/(1+u²) is the chemotactic sensitivity function.

    Components:
        - u is the cell density
        - v is the chemoattractant concentration
        - c is the chemotaxis coefficient
        - D is the chemoattractant diffusion ratio
        - a is the chemoattractant decay rate

    Pattern formation occurs for: 2√(aD) < c/2 - D - a

    Reference: Keller & Segel (1970), visualpde.com
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
                    default=5.0,
                    description="Chemotaxis coefficient",
                    min_value=1.0,
                    max_value=20.0,
                ),
                PDEParameter(
                    name="D",
                    default=0.5,
                    description="Chemoattractant diffusion ratio",
                    min_value=0.1,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="a",
                    default=1.0,
                    description="Chemoattractant decay rate",
                    min_value=0.1,
                    max_value=5.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="visualpde.com Keller-Segel chemotaxis",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        c = parameters.get("c", 5.0)
        D = parameters.get("D", 0.5)
        a = parameters.get("a", 1.0)

        # χ(u) = c*u/(1+u²)
        # ∇·(χ(u)∇v) = χ(u)*∇²v + χ'(u)*∇u·∇v
        # χ'(u) = c*(1-u²)/(1+u²)²
        # Simplified: use χ*laplace(v) + (∂χ/∂u)*gradient(u)·gradient(v)
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

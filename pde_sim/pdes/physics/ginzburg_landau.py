"""Complex Ginzburg-Landau equation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import PDEMetadata, PDEParameter, MultiFieldPDEPreset
from .. import register_pde


@register_pde("ginzburg-landau")
class GinzburgLandauPDE(MultiFieldPDEPreset):
    """Complex Ginzburg-Landau equation with full parameterization.

    Based on visualpde.com formulation:

        dψ/dt = (Dr + i*Di)∇²ψ + (ar + i*ai)ψ + (br + i*bi)ψ|ψ|²

    Writing ψ = u + i*v, this becomes:

        du/dt = Dr∇²u - Di∇²v + ar·u - ai·v + (br·u - bi·v)(u² + v²)
        dv/dt = Di∇²u + Dr∇²v + ar·v + ai·u + (br·v + bi·u)(u² + v²)

    Parameters:
        - Dr, Di: Real and imaginary diffusion coefficients
        - ar, ai: Real and imaginary linear coefficients
        - br, bi: Real and imaginary nonlinear coefficients

    For stability, typically need br, Dr >= 0.

    Reference: https://visualpde.com/nonlinear-physics/nls-cgl
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="ginzburg-landau",
            category="physics",
            description="Complex Ginzburg-Landau (real/imaginary form)",
            equations={
                "u": "Dr*laplace(u) - Di*laplace(v) + ar*u - ai*v + (br*u - bi*v)*(u**2 + v**2)",
                "v": "Di*laplace(u) + Dr*laplace(v) + ar*v + ai*u + (br*v + bi*u)*(u**2 + v**2)",
            },
            parameters=[
                PDEParameter(
                    name="Dr",
                    default=1.0,
                    description="Real diffusion coefficient",
                    min_value=0.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="Di",
                    default=0.0,
                    description="Imaginary diffusion coefficient",
                    min_value=-5.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="ar",
                    default=1.0,
                    description="Real linear coefficient",
                    min_value=-5.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="ai",
                    default=0.0,
                    description="Imaginary linear coefficient",
                    min_value=-5.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="br",
                    default=-1.0,
                    description="Real nonlinear coefficient (typically < 0)",
                    min_value=-5.0,
                    max_value=0.0,
                ),
                PDEParameter(
                    name="bi",
                    default=0.0,
                    description="Imaginary nonlinear coefficient",
                    min_value=-5.0,
                    max_value=5.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="https://visualpde.com/nonlinear-physics/nls-cgl",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Dr = parameters.get("Dr", 1.0)
        Di = parameters.get("Di", 0.0)
        ar = parameters.get("ar", 1.0)
        ai = parameters.get("ai", 0.0)
        br = parameters.get("br", -1.0)
        bi = parameters.get("bi", 0.0)

        # Complex Ginzburg-Landau in real/imaginary form:
        # du/dt = Dr∇²u - Di∇²v + ar·u - ai·v + (br·u - bi·v)(u² + v²)
        # dv/dt = Di∇²u + Dr∇²v + ar·v + ai·u + (br·v + bi·u)(u² + v²)
        return PDE(
            rhs={
                "u": f"{Dr}*laplace(u) - {Di}*laplace(v) + {ar}*u - {ai}*v + ({br}*u - {bi}*v)*(u**2 + v**2)",
                "v": f"{Di}*laplace(u) + {Dr}*laplace(v) + {ar}*v + {ai}*u + ({br}*v + {bi}*u)*(u**2 + v**2)",
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state with small perturbations."""
        if ic_type in ("ginzburg-landau-default", "default"):
            amplitude = ic_params.get("amplitude", 0.1)
            np.random.seed(ic_params.get("seed"))

            # Small random perturbations for real and imaginary parts
            u_data = amplitude * np.random.randn(*grid.shape)
            v_data = amplitude * np.random.randn(*grid.shape)

            u = ScalarField(grid, u_data)
            u.label = "u"
            v = ScalarField(grid, v_data)
            v.label = "v"

            return FieldCollection([u, v])

        # For other IC types, create the same IC for both fields
        base = create_initial_condition(grid, ic_type, ic_params)
        u = ScalarField(grid, base.data.copy())
        u.label = "u"
        v = ScalarField(grid, np.zeros(grid.shape))
        v.label = "v"
        return FieldCollection([u, v])

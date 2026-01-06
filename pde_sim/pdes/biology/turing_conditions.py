"""Demonstration that Turing instabilities are not enough for pattern formation.

Based on: "Turing instabilities are not enough to ensure pattern formation"
https://arxiv.org/abs/2308.15311
"""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("turing-conditions")
class TuringConditionsPDE(MultiFieldPDEPreset):
    """Demonstration that Turing instabilities are not enough for patterns.

    This system satisfies Turing instability conditions but the patterns
    are transient - they eventually decay to a uniform equilibrium.

        du/dt = laplace(u) + u - v - e*u³
        dv/dt = D * laplace(v) + a*v*(v + c)*(v - d) + b*u - e*v³

    The system has multiple equilibria. Turing instability around one
    equilibrium can lead to transient patterns that eventually settle
    to a different stable uniform state.

    Reference: "Turing instabilities are not enough to ensure pattern formation"
    arXiv:2308.15311, visualpde.com
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="turing-conditions",
            category="biology",
            description="Transient Turing patterns (instability not enough)",
            equations={
                "u": "laplace(u) + u - v - e * u**3",
                "v": "D * laplace(v) + a * v * (v + c) * (v - d) + b * u - e * v**3",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=30.0,
                    description="Inhibitor diffusion coefficient",
                    min_value=10.0,
                    max_value=100.0,
                ),
                PDEParameter(
                    name="a",
                    default=1.75,
                    description="Kinetic parameter a",
                    min_value=0.5,
                    max_value=3.0,
                ),
                PDEParameter(
                    name="b",
                    default=18.0,
                    description="Coupling strength u to v",
                    min_value=5.0,
                    max_value=30.0,
                ),
                PDEParameter(
                    name="c",
                    default=2.0,
                    description="Kinetic parameter c",
                    min_value=0.5,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="d",
                    default=5.0,
                    description="Kinetic parameter d",
                    min_value=1.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="e",
                    default=0.02,
                    description="Cubic damping coefficient",
                    min_value=0.01,
                    max_value=0.1,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="arXiv:2308.15311 - Turing instabilities not enough",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 30.0)
        a = parameters.get("a", 1.75)
        b = parameters.get("b", 18.0)
        c = parameters.get("c", 2.0)
        d = parameters.get("d", 5.0)
        e = parameters.get("e", 0.02)

        return PDE(
            rhs={
                "u": f"laplace(u) + u - v - {e} * u**3",
                "v": f"{D} * laplace(v) + {a} * v * (v + {c}) * (v - {d}) + {b} * u - {e} * v**3",
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create small random perturbation to trigger Turing instability."""
        np.random.seed(ic_params.get("seed"))
        amplitude = ic_params.get("amplitude", 0.05)

        # Small random perturbations around zero
        u_data = amplitude * np.random.randn(*grid.shape)
        v_data = amplitude * np.random.randn(*grid.shape)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

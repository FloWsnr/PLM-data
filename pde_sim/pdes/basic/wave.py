"""Wave equation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("wave")
class WavePDE(MultiFieldPDEPreset):
    """Wave equation.

    The wave equation describes wave propagation:

        d²u/dt² = c² * laplace(u)

    We convert to first order system:
        du/dt = v
        dv/dt = c² * laplace(u)

    where u is displacement and v is velocity.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="wave",
            category="basic",
            description="Wave equation (second order)",
            equations={
                "u": "v",
                "v": "c**2 * laplace(u)",
            },
            parameters=[
                PDEParameter(
                    name="c",
                    default=1.0,
                    description="Wave speed",
                    min_value=0.01,
                    max_value=10.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Hyperbolic PDE",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        c = parameters.get("c", 1.0)
        c_sq = c * c

        return PDE(
            rhs={
                "u": "v",
                "v": f"{c_sq} * laplace(u)",
            },
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state for wave equation."""
        from pde_sim.initial_conditions import create_initial_condition

        # u gets the specified initial condition
        u = create_initial_condition(grid, ic_type, ic_params)
        u.label = "u"

        # v (velocity) starts at zero by default
        v_data = np.zeros(grid.shape)
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])


@register_pde("advection")
class AdvectionPDE(MultiFieldPDEPreset):
    """Advection-diffusion equation.

    Describes transport of a quantity by flow and diffusion:

        du/dt = D * laplace(u) - v · grad(u)

    where D is diffusion and v is velocity vector.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="advection",
            category="basic",
            description="Advection-diffusion equation",
            equations={
                "u": "D * laplace(u) - vx * d_dx(u) - vy * d_dy(u)",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.01,
                    description="Diffusion coefficient",
                    min_value=0.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="vx",
                    default=1.0,
                    description="Advection velocity in x",
                    min_value=-10.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="vy",
                    default=0.0,
                    description="Advection velocity in y",
                    min_value=-10.0,
                    max_value=10.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Convection-diffusion equation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 0.01)
        vx = parameters.get("vx", 1.0)
        vy = parameters.get("vy", 0.0)

        # Build equation string
        terms = []
        if D > 0:
            terms.append(f"{D} * laplace(u)")
        if vx != 0:
            terms.append(f"- {vx} * d_dx(u)")
        if vy != 0:
            terms.append(f"- {vy} * d_dy(u)")

        rhs = " ".join(terms) if terms else "0"

        return PDE(
            rhs={"u": rhs},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        from pde_sim.initial_conditions import create_initial_condition
        return create_initial_condition(grid, ic_type, ic_params)

"""Advection-diffusion equation with uniform flow."""

from typing import Any

from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("advection")
class AdvectionPDE(ScalarPDEPreset):
    """Advection-diffusion equation with uniform velocity field.

    Based on visualpde.com advection equation:

        du/dt = D * laplace(u) + vx * d_dx(u) + vy * d_dy(u)

    The velocity field (vx, vy) is uniform across the domain.
    Typically used with periodic boundary conditions.

    Parameters:
        D: Diffusion coefficient
        vx: x-component of velocity
        vy: y-component of velocity
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="advection",
            category="basic",
            description="Advection-diffusion equation with uniform flow",
            equations={
                "u": "D * laplace(u) + vx * d_dx(u) + vy * d_dy(u)",
            },
            parameters=[
                PDEParameter("D", "Diffusion coefficient"),
                PDEParameter("vx", "x-component of velocity"),
                PDEParameter("vy", "y-component of velocity"),
            ],
            num_fields=1,
            field_names=["u"],
            reference="https://visualpde.com/basic-pdes/advection-equation",
            supported_dimensions=[2],  # 2D only: uses d_dy
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        vx = parameters.get("vx", 5.0)
        vy = parameters.get("vy", 0.0)

        bc_spec = self._convert_bc(bc)

        # Build advection term
        advection = f"{vx} * d_dx(u) + {vy} * d_dy(u)"

        # Build equation string
        if D > 0:
            rhs = f"{D} * laplace(u) + {advection}"
        else:
            rhs = advection

        return PDE(
            rhs={"u": rhs},
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> ScalarField:
        return create_initial_condition(grid, ic_type, ic_params)

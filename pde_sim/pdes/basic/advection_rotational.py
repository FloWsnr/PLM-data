"""Advection-diffusion equation with rotational (vortex) flow."""

from typing import Any

from pde import PDE, CartesianGrid, ScalarField

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("advection-rotational")
class AdvectionRotationalPDE(ScalarPDEPreset):
    """Advection-diffusion equation with rotational velocity field.

    Based on visualpde.com advection equation:

        du/dt = D * laplace(u) + omega * ((y - L_y/2) * d_dx(u) - (x - L_x/2) * d_dy(u))

    The velocity field rotates around the domain center:
    - Positive omega: counterclockwise rotation
    - Negative omega: clockwise rotation

    Typically used with Dirichlet boundary conditions (absorbing edges).

    Parameters:
        D: Diffusion coefficient
        omega: Angular velocity (positive = counterclockwise)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="advection-rotational",
            category="basic",
            description="Advection-diffusion equation with rotational (vortex) flow",
            equations={
                "u": "D * laplace(u) + omega * ((y - L_y/2) * d_dx(u) - (x - L_x/2) * d_dy(u))",
            },
            parameters=[
                PDEParameter("D", "Diffusion coefficient"),
                PDEParameter("omega", "Angular velocity (positive = counterclockwise)"),
            ],
            num_fields=1,
            field_names=["u"],
            reference="https://visualpde.com/basic-pdes/advection-equation",
            supported_dimensions=[2],  # 2D only: uses x, y coordinates
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        omega = parameters.get("omega", 0.1)

        # Get domain center
        L_x = grid.axes_bounds[0][1] - grid.axes_bounds[0][0]
        L_y = grid.axes_bounds[1][1] - grid.axes_bounds[1][0]
        half_x = L_x / 2
        half_y = L_y / 2

        bc_spec = self._convert_bc(bc)

        # Rotational velocity field (counterclockwise for positive omega)
        advection = f"{omega} * ((y - {half_y}) * d_dx(u) - (x - {half_x}) * d_dy(u))"

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
        return super().create_initial_state(grid, ic_type, ic_params, **kwargs)

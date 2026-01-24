"""Damped wave equation with explicit velocity damping."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("damped-wave")
class DampedWavePDE(MultiFieldPDEPreset):
    """Damped wave equation with explicit velocity damping.

    Unlike the standard wave equation which has damping via the C term in the
    displacement equation, this variant has explicit velocity damping (-d*v):

        du/dt = v + C*D * laplace(u)
        dv/dt = D * laplace(u) - d*v

    where:
        - u is displacement
        - v is velocity (du/dt)
        - D is the wave speed squared (c^2)
        - C is a stabilization parameter
        - d is the velocity damping coefficient

    This produces physically different behavior from the standard wave equation.
    The -d*v term represents energy dissipation proportional to velocity,
    similar to viscous damping in mechanical systems.

    Reference:
        https://visualpde.com/basic-pdes/damped-wave-equation
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="damped-wave",
            category="basic",
            description="Wave equation with explicit velocity damping",
            equations={
                "u": "v + C*D * laplace(u)",
                "v": "D * laplace(u) - d*v",
            },
            parameters=[
                PDEParameter("D", "Wave speed squared (c^2)"),
                PDEParameter("C", "Stabilization coefficient"),
                PDEParameter("d", "Velocity damping coefficient"),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="https://visualpde.com/basic-pdes/damped-wave-equation",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        C = parameters.get("C", 0.01)
        d = parameters.get("d", 0.0)

        bc_spec = self._convert_bc(bc)

        # Build u equation with optional stabilization term
        if C > 0:
            u_rhs = f"v + {C * D} * laplace(u)"
        else:
            u_rhs = "v"

        # Build v equation with velocity damping
        if d > 0:
            v_rhs = f"{D} * laplace(u) - {d} * v"
        else:
            v_rhs = f"{D} * laplace(u)"

        return PDE(
            rhs={
                "u": u_rhs,
                "v": v_rhs,
            },
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial state for damped wave equation."""
        # Handle default IC type
        if ic_type in ("default", "damped-wave-default"):
            ic_type = "gaussian-blob"
            ic_params = {
                "num_blobs": ic_params.get("num_blobs", 1),
                "amplitude": ic_params.get("amplitude", 1.0),
                "width": ic_params.get("width", 0.1),
                "seed": ic_params.get("seed"),
            }

        # u gets the specified initial condition
        u = create_initial_condition(grid, ic_type, ic_params)
        u.label = "u"

        # v (velocity) starts at zero by default
        v_data = np.zeros(grid.shape)
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

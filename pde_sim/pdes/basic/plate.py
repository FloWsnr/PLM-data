"""Plate vibration equation (biharmonic wave equation)."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("plate")
class PlatePDE(MultiFieldPDEPreset):
    """Plate vibration equation (biharmonic wave equation).

    The plate equation describes thin elastic plate vibrations using the
    biharmonic operator:

        d^2u/dt^2 = -D^2 * laplace(laplace(u)) - C * du/dt - Q

    Based on visualpde.com, reformulated using three fields for numerical stability:
        du/dt = v + D_c*D * laplace(u)
        dv/dt = -D * laplace(w) - C*v - Q
        w = D * laplace(u)  (auxiliary algebraic field)

    where:
        - u is plate displacement
        - v is velocity
        - w is an auxiliary field representing D * laplace(u)
        - D is the bending rigidity parameter
        - C is damping coefficient
        - Q is a constant load (gravity-like force)
        - D_c is a numerical stabilization parameter
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="plate",
            category="basic",
            description="Plate vibration equation (biharmonic wave)",
            equations={
                "u": "v + D_c*D * laplace(u)",
                "v": "-D * laplace(w) - C*v - Q",
                "w": "D * laplace(u)",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=10.0,
                    description="Bending rigidity parameter",
                    min_value=0.1,
                    max_value=100.0,
                ),
                PDEParameter(
                    name="Q",
                    default=0.003,
                    description="Constant load (gravity)",
                    min_value=0.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="C",
                    default=0.1,
                    description="Damping coefficient",
                    min_value=0.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="D_c",
                    default=0.1,
                    description="Numerical stabilization parameter",
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            num_fields=3,
            field_names=["u", "v", "w"],
            reference="https://visualpde.com/basic-pdes/plate-equation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the plate vibration PDE.

        Args:
            parameters: Dictionary containing 'D', 'Q', 'C', 'D_c' coefficients.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        D = parameters.get("D", 10.0)
        Q = parameters.get("Q", 0.003)
        C = parameters.get("C", 0.1)
        D_c = parameters.get("D_c", 0.1)

        bc_spec = self._convert_bc(bc)

        # Build u equation: v + D_c*D * laplace(u)
        if D_c > 0:
            u_rhs = f"v + {D_c * D} * laplace(u)"
        else:
            u_rhs = "v"

        # Build v equation: -D * laplace(w) - C*v - Q
        v_terms = [f"-{D} * laplace(w)"]
        if C > 0:
            v_terms.append(f"- {C} * v")
        if Q != 0:
            v_terms.append(f"- {Q}")
        v_rhs = " ".join(v_terms)

        # w is computed from u: w = D * laplace(u)
        # We use cross-diffusion to implement this relationship
        w_rhs = f"{D} * laplace(u)"

        return PDE(
            rhs={
                "u": u_rhs,
                "v": v_rhs,
                "w": w_rhs,
            },
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state for plate equation.

        Args:
            grid: The computational grid.
            ic_type: Type of initial condition.
            ic_params: Parameters for the initial condition.

        Returns:
            FieldCollection with u (displacement), v (velocity), and w (auxiliary) fields.
        """
        # u gets the specified initial condition
        u = create_initial_condition(grid, ic_type, ic_params)
        u.label = "u"

        # v (velocity) starts at zero
        v_data = np.zeros(grid.shape)
        v = ScalarField(grid, v_data)
        v.label = "v"

        # w starts at zero (will be computed from u during simulation)
        w_data = np.zeros(grid.shape)
        w = ScalarField(grid, w_data)
        w.label = "w"

        return FieldCollection([u, v, w])

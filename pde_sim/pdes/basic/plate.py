"""Plate vibration equation (biharmonic wave equation)."""

from typing import Any

import numpy as np
from pde import CartesianGrid, FieldCollection, ScalarField
from pde.pdes.base import PDEBase

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


class PlateEquationPDE(PDEBase):
    """Custom PDE for plate equation with algebraic w field.

    Implements:
        du/dt = v + D_c*D * laplace(u)
        dv/dt = -D * laplace(w) - C*v - Q
        w = D * laplace(u)  (algebraic constraint, computed each step)

    By substituting w into the v equation:
        dv/dt = -D^2 * laplace(laplace(u)) - C*v - Q
    """

    def __init__(self, D: float, Q: float, C: float, D_c: float, bc: dict):
        super().__init__()
        self.D = D
        self.Q = Q
        self.C = C
        self.D_c = D_c
        self.bc = bc

    def evolution_rate(self, state: FieldCollection, t: float = 0) -> FieldCollection:
        """Compute the right-hand side of the plate equation.

        Args:
            state: FieldCollection with [u, v, w] fields
            t: Current time (unused)

        Returns:
            FieldCollection with time derivatives [du/dt, dv/dt, dw/dt]
        """
        u, v, w = state

        # Compute laplacian of u
        laplace_u = u.laplace(bc=self.bc)

        # Compute w algebraically: w = D * laplace(u)
        # (We update w in place for consistency, though it's not evolved)
        w_algebraic = self.D * laplace_u

        # Compute laplacian of w (which is D * laplace(laplace(u)))
        # We need to create a ScalarField from w_algebraic to compute its laplacian
        w_field = ScalarField(u.grid, w_algebraic.data)
        laplace_w = w_field.laplace(bc=self.bc)

        # du/dt = v + D_c*D * laplace(u)
        du_dt = v.data + self.D_c * self.D * laplace_u.data

        # dv/dt = -D * laplace(w) - C*v - Q
        dv_dt = -self.D * laplace_w.data - self.C * v.data - self.Q

        # dw/dt = 0 (w is algebraic, but we need to return something)
        # Actually, we set dw/dt such that w tracks D * laplace(u)
        # To keep w = D * laplace(u), we set dw/dt = D * laplace(du/dt)
        # But simpler: just set dw/dt = 0 and reinitialize w each step
        dw_dt = np.zeros_like(w.data)

        # Create result fields
        result_u = ScalarField(u.grid, du_dt)
        result_v = ScalarField(v.grid, dv_dt)
        result_w = ScalarField(w.grid, dw_dt)

        return FieldCollection([result_u, result_v, result_w])


@register_pde("plate")
class PlatePDE(MultiFieldPDEPreset):
    """Plate vibration equation (biharmonic wave equation).

    The plate equation describes thin elastic plate vibrations using the
    biharmonic operator:

        d^2u/dt^2 = -D^2 * laplace(laplace(u)) - C * du/dt - Q

    Based on visualpde.com, reformulated using three fields for numerical stability:
        du/dt = v + D_c*D * laplace(u)
        dv/dt = -D * laplace(w) - C*v - Q
        w = D * laplace(u)  (auxiliary algebraic field, computed each step)

    where:
        - u is plate displacement
        - v is velocity
        - w is an auxiliary field representing D * laplace(u) (algebraic, not evolved)
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
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PlateEquationPDE:
        """Create the plate vibration PDE.

        Args:
            parameters: Dictionary containing 'D', 'Q', 'C', 'D_c' coefficients.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance with algebraic w field.
        """
        D = parameters.get("D", 10.0)
        Q = parameters.get("Q", 0.003)
        C = parameters.get("C", 0.1)
        D_c = parameters.get("D_c", 0.1)

        bc_spec = self._convert_bc(bc)

        return PlateEquationPDE(D=D, Q=Q, C=C, D_c=D_c, bc=bc_spec)

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        parameters: dict[str, float] | None = None,
        bc: dict[str, Any] | None = None,
    ) -> FieldCollection:
        """Create initial state for plate equation.

        Args:
            grid: The computational grid.
            ic_type: Type of initial condition.
            ic_params: Parameters for the initial condition.
            parameters: PDE parameters (needed to compute w = D * laplace(u)).
            bc: Boundary conditions (needed to compute laplacian).

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

        # w = D * laplace(u) (algebraic constraint)
        # Initialize w correctly from u
        D = parameters.get("D", 10.0) if parameters else 10.0
        bc_spec = self._convert_bc(bc) if bc else "auto_periodic_neumann"
        laplace_u = u.laplace(bc=bc_spec)
        w_data = D * laplace_u.data
        w = ScalarField(grid, w_data)
        w.label = "w"

        return FieldCollection([u, v, w])

"""Nonlinear beam equation with state-dependent stiffness."""

from typing import Any

import numpy as np
from pde import CartesianGrid, FieldCollection, ScalarField
from pde.pdes.base import PDEBase

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


class NonlinearBeamEquationPDE(PDEBase):
    """Custom PDE for nonlinear beam with correct equation structure.

    Implements:
        du/dt = -laplace(w)
        v = laplace(u)    (algebraic constraint, computed each step)
        w = E(v) * v      (algebraic constraint, computed each step)

    where E(v) = E_star + Delta_E * (1 + tanh(v/eps)) / 2

    This correctly implements the equation:
        du/dt = -laplace(E(laplace(u)) * laplace(u))

    Unlike the previous approximate implementation which incorrectly used:
        du/dt = -E(laplace(u)) * laplace(laplace(u))
    """

    def __init__(
        self, E_star: float, Delta_E: float, eps: float, bc: dict[str, Any]
    ):
        super().__init__()
        self.E_star = E_star
        self.Delta_E = Delta_E
        self.eps = eps
        self.bc = bc

    def _compute_stiffness(self, v_data: np.ndarray) -> np.ndarray:
        """Compute E(v) = E_star + Delta_E * (1 + tanh(v/eps)) / 2.

        Args:
            v_data: Curvature field data (laplace(u)).

        Returns:
            Stiffness field E(v) at each grid point.
        """
        return self.E_star + self.Delta_E * (1.0 + np.tanh(v_data / self.eps)) / 2.0

    def evolution_rate(
        self, state: FieldCollection, t: float = 0
    ) -> FieldCollection:
        """Compute the right-hand side of the nonlinear beam equation.

        Args:
            state: FieldCollection with [u, v, w] fields
            t: Current time (unused)

        Returns:
            FieldCollection with time derivatives [du/dt, dv/dt, dw/dt]
        """
        u, v, w = state

        # Step 1: Compute v = laplace(u) (curvature)
        v_algebraic = u.laplace(bc=self.bc)

        # Step 2: Compute E(v) - stiffness depends on curvature
        E_v = self._compute_stiffness(v_algebraic.data)

        # Step 3: Compute w = E(v) * v (moment)
        w_data = E_v * v_algebraic.data
        w_field = ScalarField(u.grid, w_data)

        # Step 4: Compute du/dt = -laplace(w)
        # This is the CORRECT equation: -laplace(E(v) * v)
        laplace_w = w_field.laplace(bc=self.bc)
        du_dt = -laplace_w.data

        # v and w are algebraic (not evolved in time)
        # Set their derivatives to zero
        dv_dt = np.zeros_like(v.data)
        dw_dt = np.zeros_like(w.data)

        # Create result fields
        result_u = ScalarField(u.grid, du_dt)
        result_v = ScalarField(v.grid, dv_dt)
        result_w = ScalarField(w.grid, dw_dt)

        return FieldCollection([result_u, result_v, result_w])


@register_pde("nonlinear-beam")
class NonlinearBeamPDE(MultiFieldPDEPreset):
    """Overdamped nonlinear beam equation with state-dependent stiffness.

    Correctly implements:
        du/dt = -laplace(E(v) * v)  where v = laplace(u)

    Using a 3-field cross-diffusion formulation:
        du/dt = -laplace(w)
        v = laplace(u)    (algebraic constraint)
        w = E(v) * v      (algebraic constraint)

    Stiffness function:
        E(v) = E_star + Delta_E * (1 + tanh(v/eps)) / 2

    For large curvatures (|v| >> eps):
        - Positive curvature: E approx E_star + Delta_E (stiffer)
        - Negative curvature: E approx E_star (softer)

    Applications:
        - MEMS resonators with nonlinear stiffness
        - Composite materials with varying properties
        - Biological structures (tendons, cartilage)
        - Smart materials (piezoelectrics, shape-memory alloys)

    Note: Visual PDE is strictly 1D; this extends to 2D via Laplacians.

    Reference: Euler-Bernoulli beam theory, Nayfeh & Pai (2004)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="nonlinear-beam",
            category="physics",
            description="Overdamped beam with curvature-dependent stiffness",
            equations={
                "u": "-laplace(w)",
                "v": "laplace(u)",
                "w": "E(v) * v",
                "E(v)": "E_star + Delta_E * (1 + tanh(v/eps)) / 2",
            },
            parameters=[
                PDEParameter(
                    name="E_star",
                    default=0.0001,
                    description="Baseline stiffness scale",
                    min_value=0.0001,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="Delta_E",
                    default=24.0,
                    description="Stiffness variation range",
                    min_value=0.0,
                    max_value=50.0,
                ),
                PDEParameter(
                    name="eps",
                    default=0.01,
                    description="Stiffness transition sharpness",
                    min_value=0.001,
                    max_value=1.0,
                ),
            ],
            num_fields=3,
            field_names=["u", "v", "w"],
            reference="Euler-Bernoulli beam theory",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> NonlinearBeamEquationPDE:
        """Create the Nonlinear Beam PDE.

        Args:
            parameters: Dictionary with E_star, Delta_E, eps.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured NonlinearBeamEquationPDE instance.
        """
        E_star = parameters.get("E_star", 0.0001)
        Delta_E = parameters.get("Delta_E", 24.0)
        eps = parameters.get("eps", 0.01)

        bc_spec = self._convert_bc(bc)

        return NonlinearBeamEquationPDE(
            E_star=E_star,
            Delta_E=Delta_E,
            eps=eps,
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        parameters: dict[str, float] | None = None,
        bc: dict[str, Any] | None = None,
    ) -> FieldCollection:
        """Create initial state for 3-field beam system.

        Args:
            grid: The computational grid.
            ic_type: Type of initial condition.
            ic_params: Parameters for the initial condition.
            parameters: PDE parameters (needed to compute w).
            bc: Boundary conditions (needed to compute laplacian).

        Returns:
            FieldCollection with u (displacement), v (curvature), w (moment).
        """
        # u gets the specified initial condition
        if ic_type in ("nonlinear-beam-default", "default"):
            u = self._create_default_displacement(grid, ic_params)
        else:
            u = create_initial_condition(grid, ic_type, ic_params)
        u.label = "u"

        # Get parameters for stiffness computation
        params = parameters or self.get_default_parameters()
        E_star = params.get("E_star", 0.0001)
        Delta_E = params.get("Delta_E", 24.0)
        eps = params.get("eps", 0.01)

        # v = laplace(u) (curvature)
        bc_spec = self._convert_bc(bc) if bc else "auto_periodic_neumann"
        v_data = u.laplace(bc=bc_spec).data
        v = ScalarField(grid, v_data)
        v.label = "v"

        # w = E(v) * v (moment)
        E_v = E_star + Delta_E * (1.0 + np.tanh(v_data / eps)) / 2.0
        w_data = E_v * v_data
        w = ScalarField(grid, w_data)
        w.label = "w"

        return FieldCollection([u, v, w])

    def _create_default_displacement(
        self, grid: CartesianGrid, ic_params: dict[str, Any]
    ) -> ScalarField:
        """Create default sinusoidal initial displacement.

        Args:
            grid: The computational grid.
            ic_params: Parameters including seed and amplitude.

        Returns:
            Initial displacement field.
        """
        seed = ic_params.get("seed")
        if seed is not None:
            np.random.seed(seed)
        amplitude = ic_params.get("amplitude", 0.5)

        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[0]

        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Sinusoidal initial shape
        data = amplitude * np.sin(np.pi * (X - x_bounds[0]) / Lx) * np.sin(
            np.pi * (Y - y_bounds[0]) / Ly
        )
        data += 0.05 * np.random.randn(*grid.shape)

        return ScalarField(grid, data)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        E_star = parameters.get("E_star", 0.0001)
        Delta_E = parameters.get("Delta_E", 24.0)
        eps = parameters.get("eps", 0.01)

        return {
            "u": "-laplace(w)",
            "v": "laplace(u)",
            "w": f"({E_star} + {Delta_E} * (1 + tanh(v / {eps})) / 2) * v",
        }

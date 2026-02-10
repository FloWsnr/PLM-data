"""Inviscid Burgers' equation for shock wave formation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("inviscid-burgers")
class InviscidBurgersPDE(ScalarPDEPreset):
    """Inviscid Burgers' equation (conservation form).

    The inviscid Burgers' equation in conservation form:

        du/dt = -d_dx(u^2/2)   (conservation form)
              = -u * d_dx(u)   (non-conservative form)

    This is the limit epsilon -> 0 of the viscous Burgers equation.
    Shocks form in finite time from smooth initial data.

    Visual PDE uses a flux-splitting formulation with algebraic auxiliary
    variable v = u^p for numerical stability:

        du/dt = -(1/(p*u^(p-2))) * d_dx(v)
        v = u^p  (algebraic)

    With p=4 and applying the chain rule, this reduces to the standard form.
    The flux-splitting helps with shock capturing in their implementation.

    Key phenomena:
        - Wave steepening: larger amplitudes travel faster
        - Shock formation: discontinuities form in finite time
        - Entropy solutions: weak solutions selected by vanishing viscosity
        - Shock interaction: shocks merge when they collide

    WARNING: Without viscosity, numerical instabilities may occur.
    Use small timesteps (dt ~ 0.001) and consider adding tiny viscosity
    (epsilon ~ 1e-4) for numerical regularization.

    Reference: Burgers (1948), Lax (1957)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="inviscid-burgers",
            category="physics",
            description="Inviscid Burgers' equation (shock formation)",
            equations={
                "u": "-u * d_dx(u)",
            },
            parameters=[
                PDEParameter("epsilon", "Viscosity (0 for truly inviscid, small for regularization)"),
                PDEParameter("direction", "Advection axis: 'x' or 'y' (default 'x')"),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Burgers (1948), Lax (1957) entropy conditions",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the inviscid Burgers' equation PDE.

        By default epsilon=0 (truly inviscid). A small epsilon can be
        added for numerical regularization if needed.
        """
        epsilon = parameters.get("epsilon", 0.0)
        direction = parameters.get("direction", "x")
        grad_op = f"d_d{direction}"

        if epsilon > 0:
            rhs = f"-u * {grad_op}(u) + {epsilon} * laplace(u)"
        else:
            rhs = f"-u * {grad_op}(u)"

        return PDE(
            rhs={"u": rhs},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> ScalarField:
        """Create initial state for inviscid Burgers' equation.

        Default: Gaussian pulse matching Visual PDE reference.
        The small offset (0.001) prevents issues at u=0.
        """
        if ic_type in ("inviscid-burgers-default", "default"):
            # Matches Visual PDE: u = 0.001 + 0.1*exp(-0.00005*(x-L_x/5)^2)
            x_bounds = grid.axes_bounds[0]
            Lx = x_bounds[1] - x_bounds[0]
            x0 = x_bounds[0] + Lx / 5  # Pulse at x = L/5

            offset = ic_params.get("offset", 0.001)
            amplitude = ic_params.get("amplitude", 0.1)
            # width_coeff corresponds to 0.00005 in Visual PDE formula
            width_coeff = ic_params.get("width_coeff", 0.00005)
            seed = ic_params.get("seed")
            if seed is not None:
                np.random.seed(seed)

            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            if len(grid.shape) > 1:
                y_bounds = grid.axes_bounds[1]
                y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
                X, Y = np.meshgrid(x, y, indexing="ij")
                data = offset + amplitude * np.exp(-width_coeff * (X - x0) ** 2)
            else:
                data = offset + amplitude * np.exp(-width_coeff * (x - x0) ** 2)

            return ScalarField(grid, data)

        if ic_type == "shock-interaction":
            # Multiple Gaussians for shock interaction demo
            return self._shock_interaction_ic(grid, ic_params)

        return create_initial_condition(grid, ic_type, ic_params)

    def _shock_interaction_ic(
        self,
        grid: CartesianGrid,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial condition with multiple shocks for interaction demo.

        Matches Visual PDE inviscidBurgersShock preset.
        """
        x_bounds = grid.axes_bounds[0]
        Lx = x_bounds[1] - x_bounds[0]

        offset = ic_params.get("offset", 0.001)
        width_coeff = ic_params.get("width_coeff", 0.001)
        seed = ic_params.get("seed")
        if seed is not None:
            np.random.seed(seed)

        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        if len(grid.shape) > 1:
            y_bounds = grid.axes_bounds[1]
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")
            x_coord = X
        else:
            x_coord = x

        # Multiple Gaussians at different positions with different amplitudes
        # Taller shocks travel faster and overtake shorter ones
        data = offset
        gaussians = [
            (Lx / 8, 0.10),       # Position, amplitude
            (Lx / 4, 0.08),
            (3 * Lx / 8, 0.06),
            (Lx / 2, 0.04),
        ]

        for pos, amp in gaussians:
            x0 = x_bounds[0] + pos
            data = data + amp * np.exp(-width_coeff * (x_coord - x0) ** 2)

        return ScalarField(grid, data)

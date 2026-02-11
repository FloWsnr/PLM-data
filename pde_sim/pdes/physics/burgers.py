"""Burgers' equation for shock wave formation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("burgers")
class BurgersPDE(ScalarPDEPreset):
    """Burgers' equation (viscous).

    Based on visualpde.com formulation:

        du/dt = -u * d_dx(u) + epsilon * laplace(u)

    The simplest nonlinear PDE combining advection and diffusion.

    Key phenomena:
        - Wave steepening: larger amplitudes travel faster
        - Shock formation: in inviscid limit, discontinuities form
        - Shock thickness: epsilon determines shock width
        - Shock interaction: multiple shocks can merge

    The viscous Burgers equation is exactly solvable via Cole-Hopf transformation.

    Reference: Burgers (1948), Cole (1951), Hopf (1950)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="burgers",
            category="physics",
            description="Burgers' equation for shock wave formation",
            equations={
                "u": "-u * d_dx(u) + epsilon * laplace(u)",
            },
            parameters=[
                PDEParameter("epsilon", "Viscosity coefficient"),
                PDEParameter("direction", "Advection axis: 'x' or 'y' (default 'x')"),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Burgers (1948) Mathematical model for turbulence",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the Burgers' equation PDE.

        Args:
            parameters: Dictionary with epsilon.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        epsilon = parameters.get("epsilon", 0.05)
        direction = parameters.get("direction", "x")

        # Viscous Burgers' equation:
        # du/dt = -u * d_d{direction}(u) + epsilon * laplace(u)
        grad_op = f"d_d{direction}"
        return PDE(
            rhs={"u": f"-u * {grad_op}(u) + {epsilon} * laplace(u)"},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> ScalarField:
        """Create initial state for Burgers' equation.

        Default: Gaussian pulse for shock formation demonstration.
        Multi-pulse: Multiple Gaussian pulses at specified x positions.
        """
        if ic_type in ("burgers-default", "default"):
            # Gaussian pulse
            x_bounds = grid.axes_bounds[0]
            Lx = x_bounds[1] - x_bounds[0]

            amplitude = ic_params.get("amplitude", 1.0)
            width = ic_params.get("width", 0.1) * Lx

            # Pulse position (randomize if not specified)
            position = ic_params.get("position")
            if position is None or position == "random":
                raise ValueError("burgers default requires position (or random)")
            x0 = x_bounds[0] + position * Lx

            ndim = len(grid.shape)
            x_1d = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            shape = [1] * ndim
            shape[0] = grid.shape[0]
            X = x_1d.reshape(shape)

            # Gaussian depends only on x, broadcast to full grid
            data = np.broadcast_to(amplitude * np.exp(-((X - x0) ** 2) / (2 * width**2)), grid.shape).copy()

            return ScalarField(grid, data)

        if ic_type == "multi-pulse":
            # Multiple Gaussian pulses at specified x positions
            # For demonstrating shock interaction (taller = faster)
            x_bounds = grid.axes_bounds[0]
            Lx = x_bounds[1] - x_bounds[0]

            # Get pulse parameters (lists)
            positions = ic_params.get("positions")
            if positions is None or positions == "random":
                raise ValueError("burgers multi-pulse requires positions (or random)")
            amplitudes = ic_params.get("amplitudes", [2.0, 1.5, 1.0])
            width = ic_params.get("width", 0.05) * Lx

            ndim = len(grid.shape)
            x_1d = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            shape = [1] * ndim
            shape[0] = grid.shape[0]
            X = x_1d.reshape(shape)

            data = np.zeros(grid.shape)
            for pos, amp in zip(positions, amplitudes):
                x0 = x_bounds[0] + pos * Lx
                data += np.broadcast_to(amp * np.exp(-((X - x0) ** 2) / (2 * width**2)), grid.shape)

            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)

    def get_position_params(self, ic_type: str) -> set[str]:
        if ic_type in ("burgers-default", "default"):
            return {"position"}
        if ic_type == "multi-pulse":
            return {"positions"}
        return super().get_position_params(ic_type)

    def resolve_ic_params(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> dict[str, Any]:
        if ic_type in ("burgers-default", "default"):
            resolved = ic_params.copy()
            if "position" not in resolved:
                raise ValueError("burgers default requires position (or random)")
            if resolved["position"] == "random":
                rng = np.random.default_rng(resolved.get("seed"))
                resolved["position"] = rng.uniform(0.1, 0.5)
            if resolved["position"] is None:
                raise ValueError("burgers default requires position (or random)")
            return resolved
        if ic_type == "multi-pulse":
            resolved = ic_params.copy()
            if "positions" not in resolved:
                raise ValueError("burgers multi-pulse requires positions (or random)")
            if resolved["positions"] == "random":
                rng = np.random.default_rng(resolved.get("seed"))
                if "amplitudes" in resolved:
                    count = len(resolved["amplitudes"])
                else:
                    count = 3
                resolved["positions"] = sorted(
                    rng.uniform(0.1, 0.6, size=count).tolist()
                )
            if resolved["positions"] is None:
                raise ValueError("burgers multi-pulse requires positions (or random)")
            return resolved
        return super().resolve_ic_params(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        epsilon = parameters.get("epsilon", 0.05)
        direction = parameters.get("direction", "x")

        return {
            "u": f"-u * d_d{direction}(u) + {epsilon} * laplace(u)",
        }

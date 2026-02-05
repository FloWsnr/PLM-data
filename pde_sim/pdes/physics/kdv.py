"""Korteweg-de Vries (KdV) equation - classical soliton equation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import PDEMetadata, PDEParameter, ScalarPDEPreset
from .. import register_pde


@register_pde("kdv")
class KdVPDE(ScalarPDEPreset):
    """Korteweg-de Vries (KdV) equation - the canonical soliton equation.

    The KdV equation:

        du/dt = -d³u/dx³ - 6*u*du/dx - b*d⁴u/dx⁴

    This is the foundational equation of soliton theory, describing weakly
    nonlinear shallow water waves, ion-acoustic waves in plasmas, and
    many other dispersive wave phenomena.

    Key properties:
        - Completely integrable (infinite conservation laws)
        - Exact N-soliton solutions via inverse scattering
        - Balance of dispersion (u_xxx) and nonlinearity (u*u_x)
        - Soliton amplitude determines speed: taller = faster

    This is a true 1D equation. The optional biharmonic dissipation term
    helps reduce numerical artifacts.

    Initial conditions:
        - Single soliton: sech²(k(x-x0)) with amplitude 2k²
        - Two solitons: superposition for collision dynamics
        - N-wave: initial bump that evolves into multiple solitons

    Reference: Korteweg & de Vries (1895), Gardner et al. (1967)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="kdv",
            category="physics",
            description="Korteweg-de Vries equation for soliton dynamics",
            equations={
                "u": "-d_dx(d_dx(d_dx(u))) - 6 * u * d_dx(u) - b * d_dx(d_dx(d_dx(d_dx(u))))",
            },
            parameters=[
                PDEParameter("b", "Biharmonic dissipation coefficient (stabilization)"),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Korteweg & de Vries (1895), Gardner et al. (1967)",
            supported_dimensions=[1],  # KdV is a true 1D equation
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        """Create the KdV equation PDE.

        Args:
            parameters: Dictionary with dissipation coefficient b.
            bc: Boundary condition specification.
            grid: The computational grid (must be 1D).

        Returns:
            Configured PDE instance.
        """
        b = parameters.get("b", 0.0001)

        # KdV equation with optional biharmonic dissipation:
        # du/dt = -d³u/dx³ - 6*u*du/dx - b*d⁴u/dx⁴
        #
        # Using py-pde's d_dx operator for derivatives.
        # For 1D, we use d_dx operators (fourth derivative for dissipation)
        if b > 0:
            rhs = f"-d_dx(d_dx(d_dx(u))) - 6 * u * d_dx(u) - {b} * d_dx(d_dx(d_dx(d_dx(u))))"
        else:
            rhs = "-d_dx(d_dx(d_dx(u))) - 6 * u * d_dx(u)"

        return PDE(
            rhs={"u": rhs},
            bc=self._convert_bc(bc, ndim=1),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> ScalarField:
        """Create initial state for KdV equation (1D only).

        Supported initial conditions:
            - default/soliton: Single sech² soliton
            - two-solitons: Two solitons for collision demonstration
            - n-wave: Initial bump that breaks into solitons
            - random-solitons: Multiple randomly placed solitons
        """
        x_bounds = grid.axes_bounds[0]
        Lx = x_bounds[1] - x_bounds[0]

        # 1D grid coordinates
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])

        if ic_type in ("kdv-default", "default", "soliton"):
            # Single soliton: u = 2k² sech²(k(x-x0))
            # Speed c = 4k², so amplitude = c/2
            k = ic_params.get("k", 0.5)  # Width parameter
            x0 = ic_params.get("x0")
            if x0 is None or x0 == "random":
                raise ValueError("kdv soliton requires x0 (or random)")

            amplitude = 2 * k**2
            data = amplitude / np.cosh(k * (x - x0)) ** 2

            return ScalarField(grid, data)

        if ic_type == "two-solitons":
            # Two solitons for collision demonstration
            # Taller soliton (larger k) moves faster and will overtake
            k1 = ic_params.get("k1", 0.6)  # Taller, faster soliton
            k2 = ic_params.get("k2", 0.4)  # Shorter, slower soliton
            x1 = ic_params.get("x1")
            x2 = ic_params.get("x2")
            if x1 is None or x1 == "random" or x2 is None or x2 == "random":
                raise ValueError("kdv two-solitons requires x1 and x2 (or random)")

            amp1 = 2 * k1**2
            amp2 = 2 * k2**2

            soliton1 = amp1 / np.cosh(k1 * (x - x1)) ** 2
            soliton2 = amp2 / np.cosh(k2 * (x - x2)) ** 2

            data = soliton1 + soliton2

            return ScalarField(grid, data)

        if ic_type == "n-wave":
            # Initial bump that evolves into multiple solitons
            # A smooth initial pulse breaks into a train of solitons
            amplitude = ic_params.get("amplitude", 1.0)
            width = ic_params.get("width", 0.1) * Lx
            x0 = ic_params.get("x0")
            if x0 is None or x0 == "random":
                raise ValueError("kdv n-wave requires x0 (or random)")

            # Gaussian bump
            data = amplitude * np.exp(-((x - x0) ** 2) / (2 * width**2))

            return ScalarField(grid, data)

        if ic_type == "offset-soliton":
            # Soliton offset from center for observing propagation
            k = ic_params.get("k", 0.5)
            x0 = ic_params.get("x0")
            if x0 is None or x0 == "random":
                raise ValueError("kdv offset-soliton requires x0 (or random)")

            amplitude = 2 * k**2
            data = amplitude / np.cosh(k * (x - x0)) ** 2

            return ScalarField(grid, data)

        if ic_type == "random-solitons":
            # N randomly placed solitons with random widths
            positions = ic_params.get("positions")
            k_values = ic_params.get("k_values")
            if (
                positions is None or positions == "random"
                or k_values is None or k_values == "random"
            ):
                raise ValueError("kdv random-solitons requires positions and k_values (or random)")

            data = np.zeros_like(x)
            for xi, ki in zip(positions, k_values):
                ampi = 2 * ki**2
                data += ampi / np.cosh(ki * (x - xi)) ** 2

            return ScalarField(grid, data)

        # Fall back to generic initial condition
        return create_initial_condition(grid, ic_type, ic_params)

    def get_position_params(self, ic_type: str) -> set[str]:
        if ic_type in ("kdv-default", "default", "soliton"):
            return {"x0"}
        if ic_type == "two-solitons":
            return {"x1", "x2"}
        if ic_type == "n-wave":
            return {"x0"}
        if ic_type == "offset-soliton":
            return {"x0"}
        if ic_type == "random-solitons":
            return {"positions"}
        return super().get_position_params(ic_type)

    def resolve_ic_params(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> dict[str, Any]:
        if ic_type in ("kdv-default", "default", "soliton"):
            resolved = ic_params.copy()
            if "x0" not in resolved:
                raise ValueError("kdv soliton requires x0 (or random)")
            if resolved["x0"] == "random":
                rng = np.random.default_rng(resolved.get("seed"))
                x_bounds = grid.axes_bounds[0]
                resolved["x0"] = rng.uniform(x_bounds[0], x_bounds[1])
            if resolved["x0"] is None or resolved["x0"] == "random":
                raise ValueError("kdv soliton requires x0 (or random)")
            return resolved

        if ic_type == "two-solitons":
            resolved = ic_params.copy()
            required = ["x1", "x2"]
            for key in required:
                if key not in resolved:
                    raise ValueError("kdv two-solitons requires x1 and x2 (or random)")
            if any(resolved[key] == "random" for key in required):
                rng = np.random.default_rng(resolved.get("seed"))
                x_bounds = grid.axes_bounds[0]
                if resolved["x1"] == "random":
                    resolved["x1"] = rng.uniform(x_bounds[0], x_bounds[1])
                if resolved["x2"] == "random":
                    resolved["x2"] = rng.uniform(x_bounds[0], x_bounds[1])
            if any(resolved[key] is None or resolved[key] == "random" for key in required):
                raise ValueError("kdv two-solitons requires x1 and x2 (or random)")
            return resolved

        if ic_type == "n-wave":
            resolved = ic_params.copy()
            if "x0" not in resolved:
                raise ValueError("kdv n-wave requires x0 (or random)")
            if resolved["x0"] == "random":
                rng = np.random.default_rng(resolved.get("seed"))
                x_bounds = grid.axes_bounds[0]
                resolved["x0"] = rng.uniform(x_bounds[0], x_bounds[1])
            if resolved["x0"] is None or resolved["x0"] == "random":
                raise ValueError("kdv n-wave requires x0 (or random)")
            return resolved

        if ic_type == "offset-soliton":
            resolved = ic_params.copy()
            if "x0" not in resolved:
                raise ValueError("kdv offset-soliton requires x0 (or random)")
            if resolved["x0"] == "random":
                rng = np.random.default_rng(resolved.get("seed"))
                x_bounds = grid.axes_bounds[0]
                resolved["x0"] = rng.uniform(x_bounds[0], x_bounds[1])
            if resolved["x0"] is None or resolved["x0"] == "random":
                raise ValueError("kdv offset-soliton requires x0 (or random)")
            return resolved

        if ic_type == "random-solitons":
            resolved = ic_params.copy()
            if "positions" not in resolved or "k_values" not in resolved:
                raise ValueError("kdv random-solitons requires positions and k_values (or random)")
            if resolved["positions"] == "random" or resolved["k_values"] == "random":
                if "n" not in resolved or "k_min" not in resolved or "k_max" not in resolved or "margin" not in resolved:
                    raise ValueError("kdv random-solitons random generation requires n, k_min, k_max, margin")
                rng = np.random.default_rng(resolved.get("seed"))
                x_bounds = grid.axes_bounds[0]
                Lx = x_bounds[1] - x_bounds[0]
                x_range = (
                    x_bounds[0] + resolved["margin"] * Lx,
                    x_bounds[0] + (1 - resolved["margin"]) * Lx * 0.5,
                )
                positions = [rng.uniform(x_range[0], x_range[1]) for _ in range(resolved["n"])]
                k_values = [rng.uniform(resolved["k_min"], resolved["k_max"]) for _ in range(resolved["n"])]
                resolved["positions"] = positions
                resolved["k_values"] = k_values
            if resolved["positions"] is None or resolved["k_values"] is None:
                raise ValueError("kdv random-solitons requires positions and k_values (or random)")
            return resolved
        return super().resolve_ic_params(grid, ic_type, ic_params)

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        b = parameters.get("b", 0.0001)

        if b > 0:
            return {
                "u": f"-d³u/dx³ - 6*u*(du/dx) - {b}*d⁴u/dx⁴",
            }
        else:
            return {
                "u": "-d³u/dx³ - 6*u*(du/dx)",
            }

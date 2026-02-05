"""SIR epidemic model with spatial diffusion."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("sir")
class SIRPDEPreset(MultiFieldPDEPreset):
    """SIR epidemic model with spatial diffusion.

    The classic Susceptible-Infected-Recovered compartmental model
    extended to include spatial dynamics through diffusion:

        dS/dt = D_S * laplace(S) - beta * S * I
        dI/dt = D_I * laplace(I) + beta * S * I - gamma * I
        dR/dt = D_R * laplace(R) + gamma * I

    Key phenomena:
        - Traveling epidemic waves: Infection fronts propagate spatially
        - Wave speed: Depends on beta, gamma, and diffusion coefficients
        - Herd immunity: Wave stops when S drops below critical threshold
        - Endemic equilibrium vs. extinction depending on R0 = beta/gamma

    The basic reproduction number R0 = beta/gamma determines outbreak behavior:
        - R0 > 1: Epidemic spreads through susceptible population
        - R0 < 1: Disease dies out

    Conservation property: S + I + R = N (constant total population)
    without demographic processes (birth/death).

    References:
        Kermack & McKendrick (1927). Proc. R. Soc. Lond. A, 115(772), 700-721.
        Murray (2002). Mathematical Biology II: Spatial Models and Biomedical
            Applications, Springer.
        Noble (1974). Nature, 250(5469), 726-728. (Spatial SIR waves)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="sir",
            category="biology",
            description="SIR epidemic model with spatial diffusion",
            equations={
                "S": "D_S * laplace(S) - beta * S * I",
                "I": "D_I * laplace(I) + beta * S * I - gamma * I",
                "R": "D_R * laplace(R) + gamma * I",
            },
            parameters=[
                PDEParameter("beta", "Infection/transmission rate"),
                PDEParameter("gamma", "Recovery rate (1/gamma = infectious period)"),
                PDEParameter("D_S", "Diffusion coefficient for susceptible"),
                PDEParameter("D_I", "Diffusion coefficient for infected"),
                PDEParameter("D_R", "Diffusion coefficient for recovered"),
            ],
            num_fields=3,
            field_names=["S", "I", "R"],
            reference="Kermack & McKendrick (1927), Murray (2002)",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        beta = parameters.get("beta", 0.5)
        gamma = parameters.get("gamma", 0.1)
        D_S = parameters.get("D_S", 0.1)
        D_I = parameters.get("D_I", 0.1)
        D_R = parameters.get("D_R", 0.1)

        # SIR equations with diffusion
        S_rhs = f"{D_S} * laplace(S) - {beta} * S * I"
        I_rhs = f"{D_I} * laplace(I) + {beta} * S * I - {gamma} * I"
        R_rhs = f"{D_R} * laplace(R) + {gamma} * I"

        return PDE(
            rhs={"S": S_rhs, "I": I_rhs, "R": R_rhs},
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial state for epidemic simulation.

        Default setup: Uniform susceptible population with a small
        localized infected region, simulating introduction of disease
        at one location.

        Parameters:
            grid: Computational grid
            ic_type: Type of initial condition ("default", "wave", or standard IC types)
            ic_params: Parameters including:
                - S0: Initial susceptible fraction (default 0.99)
                - I0: Initial infected fraction in seed region (default 0.01)
                - seed_radius: Radius of initial infection seed (default 0.1 * domain)
                - noise: Noise amplitude (default 0.0)
                - seed: Random seed
        """
        # For standard IC types, use parent class implementation
        if ic_type not in ("default", "wave"):
            return super().create_initial_state(grid, ic_type, ic_params, **kwargs)

        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.0)

        # Initial fractions
        S0 = ic_params.get("S0", 0.99)
        I0_seed = ic_params.get("I0", 0.01)

        # Get domain info
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        domain_width = x_bounds[1] - x_bounds[0]

        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Seed location (default: center or left edge for wave)
        seed_x = ic_params.get("seed_x")
        seed_y = ic_params.get("seed_y")
        if seed_x is None or seed_x == "random" or seed_y is None or seed_y == "random":
            raise ValueError("sir requires seed_x and seed_y (or random)")

        seed_radius = ic_params.get("seed_radius", 0.1 * domain_width)

        # Distance from seed point
        r = np.sqrt((X - seed_x) ** 2 + (Y - seed_y) ** 2)

        # Smooth seed using tanh profile
        I_data = I0_seed * 0.5 * (1.0 - np.tanh((r - seed_radius) / (0.1 * seed_radius)))

        # S starts uniformly high except where I is seeded
        S_data = np.full(grid.shape, S0, dtype=float)
        S_data = S_data - I_data  # Conservation: reduce S where I is added

        # R starts at zero
        R_data = np.zeros(grid.shape, dtype=float)

        # Add noise if requested
        if noise > 0:
            S_data += noise * np.random.randn(*grid.shape)
            I_data += noise * np.abs(np.random.randn(*grid.shape))

        # Ensure non-negative and bounded
        S_data = np.clip(S_data, 0.0, 1.0)
        I_data = np.clip(I_data, 0.0, 1.0)
        R_data = np.clip(R_data, 0.0, 1.0)

        # Normalize to ensure S + I + R = 1 (constant total population)
        total = S_data + I_data + R_data
        S_data = S_data / total
        I_data = I_data / total
        R_data = R_data / total

        S = ScalarField(grid, S_data)
        S.label = "S"
        I = ScalarField(grid, I_data)
        I.label = "I"
        R = ScalarField(grid, R_data)
        R.label = "R"

        return FieldCollection([S, I, R])

    def get_position_params(self, ic_type: str) -> set[str]:
        if ic_type in ("default", "wave"):
            return {"seed_x", "seed_y"}
        return super().get_position_params(ic_type)

    def resolve_ic_params(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> dict[str, Any]:
        if ic_type in ("default", "wave"):
            resolved = ic_params.copy()
            required = ["seed_x", "seed_y"]
            for key in required:
                if key not in resolved:
                    raise ValueError("sir requires seed_x and seed_y (or random)")
            if any(resolved[key] == "random" for key in required):
                rng = np.random.default_rng(resolved.get("seed"))
                x_bounds = grid.axes_bounds[0]
                y_bounds = grid.axes_bounds[1]
                if resolved["seed_x"] == "random":
                    resolved["seed_x"] = rng.uniform(x_bounds[0], x_bounds[1])
                if resolved["seed_y"] == "random":
                    resolved["seed_y"] = rng.uniform(y_bounds[0], y_bounds[1])
            if any(resolved[key] is None or resolved[key] == "random" for key in required):
                raise ValueError("sir requires seed_x and seed_y (or random)")
            return resolved

        return super().resolve_ic_params(grid, ic_type, ic_params)

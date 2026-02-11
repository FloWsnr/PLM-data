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
        at one location. Works for 1D, 2D, and 3D grids.

        Parameters:
            grid: Computational grid
            ic_type: Type of initial condition ("default", "wave", or standard IC types)
            ic_params: Parameters including:
                - S0: Initial susceptible fraction
                - I0: Initial infected fraction in seed region
                - seed_radius: Radius of initial infection seed
                - noise: Noise amplitude (default 0.0)
                - seed_x: Seed location in x (required)
                - seed_y: Seed location in y (required for 2D+)
                - seed_z: Seed location in z (required for 3D)
                - seed: Random seed
        """
        # For standard IC types, use parent class implementation
        if ic_type not in ("default", "wave"):
            return super().create_initial_state(grid, ic_type, ic_params, **kwargs)

        rng = np.random.default_rng(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.0)

        # Initial fractions
        S0 = ic_params["S0"]
        I0_seed = ic_params["I0"]

        ndim = len(grid.shape)
        domain_width = grid.axes_bounds[0][1] - grid.axes_bounds[0][0]
        seed_radius = ic_params.get("seed_radius", 0.1 * domain_width)

        # Build coordinate arrays and compute distance from seed point
        seed_keys = ["seed_x", "seed_y", "seed_z"][:ndim]
        coords_1d = []
        for dim in range(ndim):
            bounds = grid.axes_bounds[dim]
            coords_1d.append(np.linspace(bounds[0], bounds[1], grid.shape[dim]))

        if ndim == 1:
            r = np.abs(coords_1d[0] - ic_params[seed_keys[0]])
        else:
            grids = np.meshgrid(*coords_1d, indexing="ij")
            r_sq = np.zeros(grid.shape, dtype=float)
            for dim in range(ndim):
                r_sq += (grids[dim] - ic_params[seed_keys[dim]]) ** 2
            r = np.sqrt(r_sq)

        # Smooth seed using tanh profile
        I_data = I0_seed * 0.5 * (1.0 - np.tanh((r - seed_radius) / (0.1 * seed_radius)))

        # S starts uniformly high except where I is seeded
        S_data = np.full(grid.shape, S0, dtype=float) - I_data

        # R starts at zero
        R_data = np.zeros(grid.shape, dtype=float)

        # Add noise if requested
        if noise > 0:
            S_data += noise * rng.standard_normal(grid.shape)
            I_data += noise * np.abs(rng.standard_normal(grid.shape))

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

    def _seed_keys_for_grid(self, grid: CartesianGrid) -> list[str]:
        """Return the required seed position keys for the grid dimensionality."""
        ndim = len(grid.shape)
        return ["seed_x", "seed_y", "seed_z"][:ndim]

    def get_position_params(self, ic_type: str) -> set[str]:
        if ic_type in ("default", "wave"):
            # Return all possible seed keys; resolve_ic_params handles dimension filtering
            return {"seed_x", "seed_y", "seed_z"}
        return super().get_position_params(ic_type)

    def resolve_ic_params(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> dict[str, Any]:
        if ic_type in ("default", "wave"):
            resolved = ic_params.copy()
            required = self._seed_keys_for_grid(grid)
            for key in required:
                if key not in resolved:
                    raise ValueError(f"sir requires {', '.join(required)}")
            needs_random = [k for k in required if resolved[k] == "random"]
            if needs_random:
                rng = np.random.default_rng(resolved.get("seed"))
                dim_map = {"seed_x": 0, "seed_y": 1, "seed_z": 2}
                for key in needs_random:
                    bounds = grid.axes_bounds[dim_map[key]]
                    resolved[key] = rng.uniform(bounds[0], bounds[1])
            for key in required:
                if resolved[key] is None or resolved[key] == "random":
                    raise ValueError(f"sir requires {', '.join(required)}")
            return resolved

        return super().resolve_ic_params(grid, ic_type, ic_params)

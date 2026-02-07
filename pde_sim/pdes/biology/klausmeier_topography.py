"""Klausmeier model on topography with gravity-driven water flow."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("klausmeier-topography")
class KlausmeierTopographyPDE(MultiFieldPDEPreset):
    """Klausmeier model on realistic terrain.

    Extended vegetation-water model with topography-driven flow:

        dw/dt = a - w - w*n^2 + Dw*laplace(w) + V*div(w*grad(T))
        dn/dt = w*n^2 - m*n + Dn*laplace(n)

    where T(x,y) is the topographic height function.

    The advection term V*div(w*grad(T)) causes water to flow from high
    to low elevation, accumulating in valleys.

    Key phenomena:
        - Valley accumulation: water collects in low-lying areas
        - Hilltop stress: vegetation stress on exposed ridges
        - Pattern disruption: irregular terrain breaks regular stripes
        - Preferential colonization: vegetation in water-rich valleys

    Note: T is stored as a third field that remains constant after
    initial setup (with brief smoothing).

    References:
        Klausmeier, C. A. (1999). Science, 284(5421), 1826-1828.
        Saco et al. (2007). Hydrol. Earth Syst. Sci., 11(6), 1717-1730.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="klausmeier-topography",
            category="biology",
            description="Klausmeier vegetation on topography",
            equations={
                "n": "Dn * laplace(n) + w * n**2 - m * n",
                "w": "Dw * laplace(w) + a - w - w * n**2 + V * div(w * gradient(T))",
                "T": "0",  # Topography is static
            },
            parameters=[
                PDEParameter("a", "Rainfall rate"),
                PDEParameter("m", "Plant mortality rate"),
                PDEParameter("V", "Gravity-driven flow strength"),
                PDEParameter("Dn", "Plant diffusion coefficient"),
                PDEParameter("Dw", "Water diffusion coefficient"),
            ],
            num_fields=3,
            field_names=["n", "w", "T"],
            reference="Klausmeier (1999), Saco et al. (2007)",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        a = parameters["a"]
        m = parameters["m"]
        V = parameters["V"]
        Dn = parameters["Dn"]
        Dw = parameters["Dw"]

        # Use conservative divergence form for topographic water flow:
        # div(w * grad(T)) is computed as a single operator for numerical stability.
        # The expanded form w*laplace(T) + inner(grad(w), grad(T)) is unstable
        # because laplace() and divergence(gradient()) use different stencils.
        return PDE(
            rhs={
                "n": f"{Dn} * laplace(n) + w * n**2 - {m} * n",
                "w": f"{Dw} * laplace(w) + {a} - w - w * n**2 + {V} * divergence(w * gradient(T))",
                "T": "0",
            },
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial state with topography.

        Topography types (set via ic_params["topography"]):
            - "slope": Simple linear slope in x direction
            - "hills": Sinusoidal hills pattern
            - "valley": Central valley with surrounding ridges
            - "ridge": Central ridge with valleys on sides
            - "random": Random smooth terrain (sum of sinusoids)
        """
        # For standard IC types, use parent class implementation
        if ic_type not in ("default", "custom"):
            return super().create_initial_state(grid, ic_type, ic_params, **kwargs)

        rng = np.random.default_rng(ic_params.get("seed"))

        # Initial water: constant (similar to visual-pde initCond_2: "1")
        w_data = np.ones(grid.shape)

        # Initial plants: random gaussian around vegetated steady state
        n_mean = ic_params["n_mean"]
        n_std = ic_params["n_std"]
        n_data = rng.normal(n_mean, n_std, grid.shape)
        n_data = np.clip(n_data, 0, None)  # Ensure non-negative

        # Build coordinate grids
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[0]
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Normalized coordinates [0, 1]
        Xn = (X - x_bounds[0]) / Lx
        Yn = (Y - y_bounds[0]) / Ly

        # Generate topography based on type
        topo_type = ic_params["topography"]
        amplitude = ic_params["amplitude"]
        base_slope = ic_params.get("base_slope", 0.0)

        if topo_type == "slope":
            slope = ic_params["slope"]
            T_data = amplitude * slope * X

        elif topo_type == "hills":
            n_hills_x = ic_params["n_hills_x"]
            n_hills_y = ic_params["n_hills_y"]
            T_data = amplitude * (
                np.sin(n_hills_x * np.pi * Xn) * np.sin(n_hills_y * np.pi * Yn)
            )

        elif topo_type == "gaussian_blobs":
            blob_width = ic_params["blob_width"]
            T_data = np.zeros_like(X)

            blob_positions = ic_params["blob_positions"]
            blob_amplitudes = ic_params["blob_amplitudes"]

            if len(blob_positions) != len(blob_amplitudes):
                raise ValueError("klausmeier-topography gaussian_blobs requires matching blob_positions and blob_amplitudes lengths")

            for (cx, cy), blob_amp in zip(blob_positions, blob_amplitudes):
                T_data += blob_amp * np.exp(
                    -((Xn - cx) ** 2 + (Yn - cy) ** 2) / (2 * blob_width**2)
                )
            T_data = amplitude * T_data

        elif topo_type == "valley":
            T_data = amplitude * (2 * Yn - 1) ** 2

        elif topo_type == "ridge":
            T_data = amplitude * (1 - (2 * Yn - 1) ** 2)

        elif topo_type == "random":
            modes = ic_params["modes"]
            T_data = np.zeros_like(X)
            for mode in modes:
                T_data += mode["amp"] * np.sin(
                    mode["kx"] * np.pi * Xn + mode["phase_x"]
                ) * np.sin(mode["ky"] * np.pi * Yn + mode["phase_y"])
            T_data = amplitude * T_data / len(modes)

        else:
            raise ValueError(f"Unknown topography type: {topo_type}")

        # Add base slope: base_slope * (x/Lx - 0.5)
        # This matches the reference: 20*(x/L_x-0.5)
        if base_slope != 0.0:
            T_data += base_slope * (Xn - 0.5)

        n = ScalarField(grid, n_data)
        n.label = "n"
        w = ScalarField(grid, w_data)
        w.label = "w"
        T = ScalarField(grid, T_data)
        T.label = "T"

        return FieldCollection([n, w, T])

    def get_position_params(self, ic_type: str) -> set[str]:
        if ic_type == "gaussian_blobs":
            return {"blob_positions"}
        return super().get_position_params(ic_type)

    def resolve_ic_params(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> dict[str, Any]:
        if ic_type not in ("default", "custom"):
            return super().resolve_ic_params(grid, ic_type, ic_params)

        resolved = ic_params.copy()
        if "topography" not in resolved:
            return resolved

        topo_type = resolved["topography"]
        if topo_type == "gaussian_blobs":
            if "blob_positions" not in resolved or "blob_amplitudes" not in resolved:
                raise ValueError("klausmeier-topography gaussian_blobs requires blob_positions and blob_amplitudes (or random)")
            if resolved["blob_positions"] == "random" or resolved["blob_amplitudes"] == "random":
                if "n_blobs" not in resolved:
                    raise ValueError("klausmeier-topography gaussian_blobs random generation requires n_blobs")
                rng = np.random.default_rng(resolved.get("seed"))
                blob_positions = [
                    [rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)]
                    for _ in range(resolved["n_blobs"])
                ]
                blob_amplitudes = [rng.uniform(0.5, 1.0) for _ in range(resolved["n_blobs"])]
                resolved["blob_positions"] = blob_positions
                resolved["blob_amplitudes"] = blob_amplitudes
            if resolved["blob_positions"] is None or resolved["blob_amplitudes"] is None:
                raise ValueError("klausmeier-topography gaussian_blobs requires blob_positions and blob_amplitudes (or random)")
            return resolved

        if topo_type == "random":
            if "modes" not in resolved:
                raise ValueError("klausmeier-topography random requires modes (or random)")
            if resolved["modes"] == "random":
                if "n_modes" not in resolved:
                    raise ValueError("klausmeier-topography random generation requires n_modes")
                rng = np.random.default_rng(resolved.get("seed"))
                modes = []
                for _ in range(resolved["n_modes"]):
                    modes.append(
                        {
                            "kx": rng.uniform(1, 4),
                            "ky": rng.uniform(1, 4),
                            "phase_x": rng.uniform(0, 2 * np.pi),
                            "phase_y": rng.uniform(0, 2 * np.pi),
                            "amp": rng.uniform(0.5, 1.0),
                        }
                    )
                resolved["modes"] = modes
            if resolved["modes"] is None:
                raise ValueError("klausmeier-topography random requires modes (or random)")
            return resolved

        return resolved

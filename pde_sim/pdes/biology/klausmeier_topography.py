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
                PDEParameter(
                    name="a",
                    default=2.0,
                    description="Rainfall rate",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="m",
                    default=0.54,
                    description="Plant mortality rate",
                    min_value=0.2,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="V",
                    default=100.0,
                    description="Gravity-driven flow strength",
                    min_value=0.0,
                    max_value=500.0,
                ),
                PDEParameter(
                    name="Dn",
                    default=1.0,
                    description="Plant diffusion coefficient",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dw",
                    default=2.0,
                    description="Water diffusion coefficient",
                    min_value=0.1,
                    max_value=10.0,
                ),
            ],
            num_fields=3,
            field_names=["n", "w", "T"],
            reference="Klausmeier (1999), Saco et al. (2007)",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        a = parameters.get("a", 2.0)
        m = parameters.get("m", 0.54)
        V = parameters.get("V", 100.0)
        Dn = parameters.get("Dn", 1.0)
        Dw = parameters.get("Dw", 2.0)

        # The div(w*grad(T)) term implements topographic water flow
        # = w*laplace(T) + inner(gradient(w), gradient(T))
        return PDE(
            rhs={
                "n": f"{Dn} * laplace(n) + w * n**2 - {m} * n",
                "w": f"{Dw} * laplace(w) + {a} - w - w * n**2 + {V} * (w * laplace(T) + inner(gradient(w), gradient(T)))",
                "T": "0",  # Topography is static
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
        np.random.seed(ic_params.get("seed"))

        # Initial water: constant (similar to visual-pde initCond_2: "1")
        w_data = np.ones(grid.shape)

        # Initial plants: random gaussian around vegetated steady state
        # Using mean=0.5, std=0.1 works better than uniform [0,1] for pattern formation
        n_mean = ic_params.get("n_mean", 0.5)
        n_std = ic_params.get("n_std", 0.1)
        n_data = np.random.normal(n_mean, n_std, grid.shape)
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
        topo_type = ic_params.get("topography", "hills")
        amplitude = ic_params.get("amplitude", 10.0)  # Default ~10 to match reference
        base_slope = ic_params.get("base_slope", 0.0)  # Linear gradient strength

        if topo_type == "slope":
            # Simple linear slope in x direction
            slope = ic_params.get("slope", 0.1)
            T_data = amplitude * slope * X

        elif topo_type == "hills":
            # Sinusoidal hills pattern
            n_hills_x = ic_params.get("n_hills_x", 3)
            n_hills_y = ic_params.get("n_hills_y", 3)
            T_data = amplitude * (
                np.sin(n_hills_x * np.pi * Xn) * np.sin(n_hills_y * np.pi * Yn)
            )

        elif topo_type == "gaussian_blobs":
            # Gaussian blob hills - more localized peaks than sinusoidal
            n_blobs = ic_params.get("n_blobs", 4)
            blob_width = ic_params.get("blob_width", 0.15)  # Width as fraction of domain
            T_data = np.zeros_like(X)

            # Generate random blob positions
            for _ in range(n_blobs):
                cx = np.random.uniform(0.1, 0.9)  # Blob center x (normalized)
                cy = np.random.uniform(0.1, 0.9)  # Blob center y (normalized)
                blob_amp = np.random.uniform(0.5, 1.0)  # Random amplitude variation
                sigma = blob_width

                # Gaussian blob
                T_data += blob_amp * np.exp(
                    -((Xn - cx) ** 2 + (Yn - cy) ** 2) / (2 * sigma**2)
                )

            # Scale to amplitude
            T_data = amplitude * T_data

        elif topo_type == "valley":
            # Central valley running in x direction (low in center, high on edges)
            T_data = amplitude * (2 * Yn - 1) ** 2

        elif topo_type == "ridge":
            # Central ridge running in x direction (high in center, low on edges)
            T_data = amplitude * (1 - (2 * Yn - 1) ** 2)

        elif topo_type == "random":
            # Random smooth terrain from sum of sinusoids
            n_modes = ic_params.get("n_modes", 5)
            T_data = np.zeros_like(X)
            for _ in range(n_modes):
                kx = np.random.uniform(1, 4)
                ky = np.random.uniform(1, 4)
                phase_x = np.random.uniform(0, 2 * np.pi)
                phase_y = np.random.uniform(0, 2 * np.pi)
                amp = np.random.uniform(0.5, 1.0)
                T_data += amp * np.sin(kx * np.pi * Xn + phase_x) * np.sin(
                    ky * np.pi * Yn + phase_y
                )
            # Normalize and scale
            T_data = amplitude * T_data / n_modes

        else:
            # Default: gentle slope
            T_data = amplitude * 0.1 * X

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

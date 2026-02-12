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

        randomize = kwargs.get("randomize", False)
        rng = np.random.default_rng(ic_params.get("seed"))

        # Initial water: constant (similar to visual-pde initCond_2: "1")
        w_data = np.ones(grid.shape)

        # Initial plants: random gaussian around vegetated steady state
        n_mean = ic_params["n_mean"]
        n_std = ic_params["n_std"]
        n_data = rng.normal(n_mean, n_std, grid.shape)
        n_data = np.clip(n_data, 0, None)  # Ensure non-negative

        # Build N-D coordinate grids
        ndim = len(grid.shape)
        coords_1d = [np.linspace(grid.axes_bounds[i][0], grid.axes_bounds[i][1], grid.shape[i]) for i in range(ndim)]
        coords = np.meshgrid(*coords_1d, indexing="ij")

        # Domain sizes
        L = [grid.axes_bounds[i][1] - grid.axes_bounds[i][0] for i in range(ndim)]

        # Normalized coordinates [0, 1]
        coords_norm = [(coords[i] - grid.axes_bounds[i][0]) / L[i] for i in range(ndim)]

        # Generate topography based on type
        topo_type = ic_params["topography"]
        amplitude = ic_params["amplitude"]
        base_slope = ic_params.get("base_slope", 0.0)

        if topo_type == "slope":
            slope = ic_params["slope"]
            T_data = amplitude * slope * coords[0]

        elif topo_type == "hills":
            # Product of sin across all available dimensions
            dim_names = ["x", "y", "z"][:ndim]
            n_hills = [ic_params[f"n_hills_{d}"] for d in dim_names]
            T_data = amplitude * np.prod(
                [np.sin(n_hills[i] * np.pi * coords_norm[i]) for i in range(ndim)],
                axis=0,
            )

        elif topo_type == "gaussian_blobs":
            blob_width = ic_params["blob_width"]
            T_data = np.zeros(grid.shape)

            blob_positions = ic_params["blob_positions"]
            blob_amplitudes = ic_params["blob_amplitudes"]

            if randomize:
                n_blobs = ic_params.get("n_blobs")
                if n_blobs is None:
                    if isinstance(blob_positions, list):
                        n_blobs = len(blob_positions)
                    elif isinstance(blob_amplitudes, list):
                        n_blobs = len(blob_amplitudes)
                    else:
                        raise ValueError("klausmeier-topography gaussian_blobs random generation requires n_blobs")
                n_blobs = int(n_blobs)
                blob_positions = [
                    [rng.uniform(0.1, 0.9) for _ in range(ndim)]
                    for _ in range(n_blobs)
                ]
                blob_amplitudes = [rng.uniform(0.5, 1.0) for _ in range(n_blobs)]

            if len(blob_positions) != len(blob_amplitudes):
                raise ValueError("klausmeier-topography gaussian_blobs requires matching blob_positions and blob_amplitudes lengths")

            for pos, blob_amp in zip(blob_positions, blob_amplitudes):
                r_sq = sum((coords_norm[i] - pos[i])**2 for i in range(ndim))
                T_data += blob_amp * np.exp(-r_sq / (2 * blob_width**2))
            T_data = amplitude * T_data

        elif topo_type == "valley":
            # Uses second axis if available, else first axis
            axis_idx = min(1, ndim - 1)
            T_data = amplitude * (2 * coords_norm[axis_idx] - 1) ** 2

        elif topo_type == "ridge":
            # Same as valley but inverted
            axis_idx = min(1, ndim - 1)
            T_data = amplitude * (1 - (2 * coords_norm[axis_idx] - 1) ** 2)

        elif topo_type == "random":
            modes = ic_params["modes"]
            dim_names = ["x", "y", "z"][:ndim]

            if randomize:
                if "n_modes" not in ic_params:
                    raise ValueError("klausmeier-topography random generation requires n_modes")
                modes = []
                for _ in range(ic_params["n_modes"]):
                    mode = {"amp": rng.uniform(0.5, 1.0)}
                    for d in dim_names:
                        mode[f"k{d}"] = rng.uniform(1, 4)
                        mode[f"phase_{d}"] = rng.uniform(0, 2 * np.pi)
                    modes.append(mode)
            T_data = np.zeros(grid.shape)
            for mode in modes:
                term = mode["amp"]
                for i, d in enumerate(dim_names):
                    term = term * np.sin(mode[f"k{d}"] * np.pi * coords_norm[i] + mode[f"phase_{d}"])
                T_data += term
            T_data = amplitude * T_data / len(modes)

        else:
            raise ValueError(f"Unknown topography type: {topo_type}")

        # Add base slope: base_slope * (x/Lx - 0.5)
        # This matches the reference: 20*(x/L_x-0.5)
        if base_slope != 0.0:
            T_data += base_slope * (coords_norm[0] - 0.5)

        n = ScalarField(grid, n_data)
        n.label = "n"
        w = ScalarField(grid, w_data)
        w.label = "w"
        T = ScalarField(grid, T_data)
        T.label = "T"

        return FieldCollection([n, w, T])


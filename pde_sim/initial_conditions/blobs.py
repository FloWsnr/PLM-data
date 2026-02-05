"""Gaussian blob initial condition generator."""

import numpy as np
from pde import CartesianGrid, ScalarField

from .base import InitialConditionGenerator


class GaussianBlob(InitialConditionGenerator):
    """Gaussian blob initial condition generator.

    Creates one or more Gaussian blobs at specified positions.
    Supports symmetric (circular/spherical) and asymmetric (elliptical/ellipsoidal) blobs.
    Works with 1D, 2D, and 3D grids.
    """

    def generate(
        self,
        grid: CartesianGrid,
        num_blobs: int = 1,
        positions: list | None = None,
        amplitude: float = 1.0,
        width: float = 0.1,
        background: float = 0.0,
        seed: int | None = None,
        random_amplitude: bool = False,
        aspect_ratio: float = 1.0,
        random_aspect: bool = False,
        **kwargs,
    ) -> ScalarField:
        """Generate Gaussian blob initial condition.

        Args:
            grid: The computational grid (1D, 2D, or 3D).
            num_blobs: Number of Gaussian blobs to create.
            positions: Blob centers in normalized coordinates (0-1).
                For 1D: [x1, x2, ...]
                For 2D: [[x1, y1], [x2, y2], ...]
                For 3D: [[x1, y1, z1], [x2, y2, z2], ...]
            amplitude: Peak amplitude of each blob (or max if random_amplitude).
            width: Width of blobs relative to domain size.
            background: Background value.
            seed: Random seed for blob placement (for reproducibility).
            random_amplitude: If True, randomize blob amplitudes in [0.5*amplitude, amplitude].
            aspect_ratio: Ratio of major to minor axis for asymmetric blobs.
                          1.0 gives symmetric blobs; >1.0 gives elongated blobs.
                          Ignored for 1D grids.
            random_aspect: If True, randomize aspect ratio for each blob in [1.5, aspect_ratio].
            **kwargs: Additional arguments (ignored).

        Returns:
            ScalarField with Gaussian blobs.
        """
        rng = np.random.default_rng(seed)
        data = np.full(grid.shape, background, dtype=float)
        ndim = len(grid.shape)

        if positions is None:
            raise ValueError("positions must be provided for gaussian-blob initial conditions")

        bounds = grid.axes_bounds
        sizes = [b[1] - b[0] for b in bounds]
        positions_array = np.array(positions, dtype=float)

        if ndim == 1:
            if positions_array.ndim != 1:
                raise ValueError("1D gaussian-blob positions must be a list of floats")
            positions_list = [float(x) for x in positions_array.tolist()]
        else:
            if positions_array.ndim != 2 or positions_array.shape[1] != ndim:
                raise ValueError(f"{ndim}D gaussian-blob positions must be a list of {ndim}D coordinates")
            positions_list = [list(row) for row in positions_array.tolist()]

        if len(positions_list) != num_blobs:
            raise ValueError(
                f"num_blobs={num_blobs} does not match positions length {len(positions_list)}"
            )

        centers = []
        for pos in positions_list:
            if ndim == 1:
                norm = float(pos)
                if not 0.0 <= norm <= 1.0:
                    raise ValueError("1D gaussian-blob position must be in [0, 1]")
                centers.append(bounds[0][0] + norm * sizes[0])
            else:
                coords = []
                for i, norm in enumerate(pos):
                    if not 0.0 <= norm <= 1.0:
                        raise ValueError("gaussian-blob positions must be in [0, 1]")
                    coords.append(bounds[i][0] + norm * sizes[i])
                centers.append(coords)

        if ndim == 1:
            self._generate_1d(
                data, grid, rng, centers, amplitude, width, random_amplitude
            )
        elif ndim == 2:
            self._generate_2d(
                data,
                grid,
                rng,
                centers,
                amplitude,
                width,
                random_amplitude,
                aspect_ratio,
                random_aspect,
            )
        else:  # 3D
            self._generate_3d(
                data,
                grid,
                rng,
                centers,
                amplitude,
                width,
                random_amplitude,
                aspect_ratio,
                random_aspect,
            )

        return ScalarField(grid, data)

    @classmethod
    def get_position_params(cls) -> set[str]:
        """Return names of parameters that represent spatial positions."""
        return {"positions"}

    @classmethod
    def resolve_random_params(
        cls,
        grid: CartesianGrid,
        params: dict,
    ) -> dict:
        """Resolve random positions for gaussian-blob."""
        resolved = params.copy()
        if "positions" not in resolved:
            raise ValueError("gaussian-blob requires positions or positions: random")
        positions = resolved["positions"]
        if positions == "random":
            if "num_blobs" not in resolved:
                raise ValueError("gaussian-blob random positions require num_blobs")
            num_blobs = resolved["num_blobs"]
            seed = resolved.get("seed")
            rng = np.random.default_rng(seed)
            ndim = len(grid.shape)
            if ndim == 1:
                resolved["positions"] = [rng.uniform(0.0, 1.0) for _ in range(num_blobs)]
            elif ndim == 2:
                resolved["positions"] = [
                    [rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)]
                    for _ in range(num_blobs)
                ]
            else:
                resolved["positions"] = [
                    [
                        rng.uniform(0.0, 1.0),
                        rng.uniform(0.0, 1.0),
                        rng.uniform(0.0, 1.0),
                    ]
                    for _ in range(num_blobs)
                ]

        return resolved

    def _generate_1d(
        self,
        data: np.ndarray,
        grid: CartesianGrid,
        rng: np.random.Generator,
        centers: list[float],
        amplitude: float,
        width: float,
        random_amplitude: bool,
    ) -> None:
        """Generate 1D symmetric Gaussian blobs."""
        x_bounds = grid.axes_bounds[0]
        Lx = x_bounds[1] - x_bounds[0]
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        sigma_x = width * Lx

        for cx in centers:
            amp = rng.uniform(0.5 * amplitude, amplitude) if random_amplitude else amplitude
            blob = amp * np.exp(-((x - cx) ** 2) / (2 * sigma_x**2))
            data += blob

    def _generate_2d(
        self,
        data: np.ndarray,
        grid: CartesianGrid,
        rng: np.random.Generator,
        centers: list[list[float]],
        amplitude: float,
        width: float,
        random_amplitude: bool,
        aspect_ratio: float,
        random_aspect: bool,
    ) -> None:
        """Generate 2D Gaussian blobs (symmetric or asymmetric)."""
        x_bounds, y_bounds = grid.axes_bounds[0], grid.axes_bounds[1]
        Lx, Ly = x_bounds[1] - x_bounds[0], y_bounds[1] - y_bounds[0]
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        for cx, cy in centers:
            amp = rng.uniform(0.5 * amplitude, amplitude) if random_amplitude else amplitude

            if aspect_ratio == 1.0 and not random_aspect:
                # Symmetric blob
                sigma_x, sigma_y = width * Lx, width * Ly
                blob = amp * np.exp(
                    -((X - cx) ** 2 / (2 * sigma_x**2) + (Y - cy) ** 2 / (2 * sigma_y**2))
                )
            else:
                # Asymmetric blob with random orientation
                theta = rng.uniform(0, 2 * np.pi)

                if random_aspect:
                    ar = rng.uniform(1.5, max(1.5, aspect_ratio))
                else:
                    ar = aspect_ratio

                sigma_minor = width * min(Lx, Ly)
                sigma_major = sigma_minor * ar

                # Rotate coordinates
                X_rot = (X - cx) * np.cos(theta) + (Y - cy) * np.sin(theta)
                Y_rot = -(X - cx) * np.sin(theta) + (Y - cy) * np.cos(theta)

                blob = amp * np.exp(
                    -(X_rot**2 / (2 * sigma_major**2) + Y_rot**2 / (2 * sigma_minor**2))
                )

            data += blob

    def _generate_3d(
        self,
        data: np.ndarray,
        grid: CartesianGrid,
        rng: np.random.Generator,
        centers: list[list[float]],
        amplitude: float,
        width: float,
        random_amplitude: bool,
        aspect_ratio: float,
        random_aspect: bool,
    ) -> None:
        """Generate 3D Gaussian blobs (symmetric or asymmetric)."""
        x_bounds, y_bounds, z_bounds = (
            grid.axes_bounds[0],
            grid.axes_bounds[1],
            grid.axes_bounds[2],
        )
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[0]
        Lz = z_bounds[1] - z_bounds[0]
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        z = np.linspace(z_bounds[0], z_bounds[1], grid.shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        for cx, cy, cz in centers:
            amp = rng.uniform(0.5 * amplitude, amplitude) if random_amplitude else amplitude

            if aspect_ratio == 1.0 and not random_aspect:
                # Symmetric blob
                sigma_x, sigma_y, sigma_z = width * Lx, width * Ly, width * Lz
                blob = amp * np.exp(
                    -(
                        (X - cx) ** 2 / (2 * sigma_x**2)
                        + (Y - cy) ** 2 / (2 * sigma_y**2)
                        + (Z - cz) ** 2 / (2 * sigma_z**2)
                    )
                )
            else:
                # Asymmetric ellipsoidal blob with random 3D orientation
                # Random unit vector for elongation direction
                phi = rng.uniform(0, 2 * np.pi)
                cos_theta = rng.uniform(-1, 1)
                sin_theta = np.sqrt(1 - cos_theta**2)
                n = np.array(
                    [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta]
                )

                if random_aspect:
                    ar = rng.uniform(1.5, max(1.5, aspect_ratio))
                else:
                    ar = aspect_ratio

                sigma_minor = width * min(Lx, Ly, Lz)
                sigma_major = sigma_minor * ar

                # Project distances onto the elongation axis and perpendicular
                d_parallel = (X - cx) * n[0] + (Y - cy) * n[1] + (Z - cz) * n[2]
                d_perp_sq = (
                    (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
                ) - d_parallel**2

                blob = amp * np.exp(
                    -(
                        d_parallel**2 / (2 * sigma_major**2)
                        + d_perp_sq / (2 * sigma_minor**2)
                    )
                )

            data += blob

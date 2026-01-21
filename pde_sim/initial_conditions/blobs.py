"""Gaussian blob initial condition generators."""

import numpy as np
from pde import CartesianGrid, ScalarField

from .base import InitialConditionGenerator


class GaussianBlobs(InitialConditionGenerator):
    """Multiple Gaussian blobs initial condition.

    Creates a field with multiple Gaussian-shaped bumps at random positions.
    Supports 1D, 2D, and 3D grids.
    """

    def generate(
        self,
        grid: CartesianGrid,
        num_blobs: int = 5,
        amplitude: float = 1.0,
        width: float = 0.1,
        background: float = 0.0,
        random_amplitude: bool = False,
        seed: int | None = None,
        **kwargs,
    ) -> ScalarField:
        """Generate Gaussian blobs initial condition.

        Args:
            grid: The computational grid (1D, 2D, or 3D).
            num_blobs: Number of Gaussian blobs to create.
            amplitude: Peak amplitude of each blob (or max if random_amplitude).
            width: Width of blobs relative to domain size.
            background: Background value.
            random_amplitude: If True, randomize blob amplitudes.
            seed: Random seed for blob placement (for reproducibility).
            **kwargs: Additional arguments (ignored).

        Returns:
            ScalarField with Gaussian blobs.
        """
        rng = np.random.default_rng(seed)
        data = np.full(grid.shape, background, dtype=float)
        ndim = len(grid.shape)

        # Get domain info
        bounds = grid.axes_bounds
        sizes = [b[1] - b[0] for b in bounds]

        if ndim == 1:
            x_bounds = bounds[0]
            Lx = sizes[0]
            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            sigma_x = width * Lx

            for _ in range(num_blobs):
                cx = rng.uniform(x_bounds[0], x_bounds[1])
                amp = rng.uniform(0.5 * amplitude, amplitude) if random_amplitude else amplitude
                blob = amp * np.exp(-((x - cx) ** 2) / (2 * sigma_x**2))
                data += blob

        elif ndim == 2:
            x_bounds, y_bounds = bounds[0], bounds[1]
            Lx, Ly = sizes[0], sizes[1]
            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")
            sigma_x, sigma_y = width * Lx, width * Ly

            for _ in range(num_blobs):
                cx = rng.uniform(x_bounds[0], x_bounds[1])
                cy = rng.uniform(y_bounds[0], y_bounds[1])
                amp = rng.uniform(0.5 * amplitude, amplitude) if random_amplitude else amplitude
                blob = amp * np.exp(
                    -((X - cx) ** 2 / (2 * sigma_x**2) + (Y - cy) ** 2 / (2 * sigma_y**2))
                )
                data += blob

        else:  # 3D
            x_bounds, y_bounds, z_bounds = bounds[0], bounds[1], bounds[2]
            Lx, Ly, Lz = sizes[0], sizes[1], sizes[2]
            x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            z = np.linspace(z_bounds[0], z_bounds[1], grid.shape[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            sigma_x, sigma_y, sigma_z = width * Lx, width * Ly, width * Lz

            for _ in range(num_blobs):
                cx = rng.uniform(x_bounds[0], x_bounds[1])
                cy = rng.uniform(y_bounds[0], y_bounds[1])
                cz = rng.uniform(z_bounds[0], z_bounds[1])
                amp = rng.uniform(0.5 * amplitude, amplitude) if random_amplitude else amplitude
                blob = amp * np.exp(
                    -((X - cx) ** 2 / (2 * sigma_x**2)
                      + (Y - cy) ** 2 / (2 * sigma_y**2)
                      + (Z - cz) ** 2 / (2 * sigma_z**2))
                )
                data += blob

        return ScalarField(grid, data)


class AsymmetricBlobs(InitialConditionGenerator):
    """Multiple asymmetric (elliptical) Gaussian blobs with random orientations.

    Creates a field with elongated Gaussian-shaped bumps at random positions
    and random orientations, useful for seeding worm-like patterns.
    """

    def generate(
        self,
        grid: CartesianGrid,
        num_blobs: int = 10,
        amplitude: float = 1.0,
        width: float = 0.02,
        aspect_ratio: float = 3.0,
        background: float = 0.0,
        random_aspect: bool = True,
        seed: int | None = None,
        **kwargs,
    ) -> ScalarField:
        """Generate asymmetric Gaussian blobs initial condition.

        Args:
            grid: The computational grid.
            num_blobs: Number of blobs to create.
            amplitude: Peak amplitude of each blob.
            width: Base width of blobs relative to domain size (minor axis).
            aspect_ratio: Ratio of major to minor axis (elongation).
            background: Background value.
            random_aspect: If True, randomize aspect ratio for each blob.
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments (ignored).

        Returns:
            ScalarField with asymmetric Gaussian blobs.
        """
        rng = np.random.default_rng(seed)
        data = np.full(grid.shape, background, dtype=float)

        # Get domain bounds
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[0]

        # Create coordinate arrays
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        for _ in range(num_blobs):
            # Random center
            cx = rng.uniform(x_bounds[0], x_bounds[1])
            cy = rng.uniform(y_bounds[0], y_bounds[1])

            # Random orientation angle
            theta = rng.uniform(0, 2 * np.pi)

            # Aspect ratio (elongation)
            if random_aspect:
                ar = rng.uniform(1.5, aspect_ratio)
            else:
                ar = aspect_ratio

            # Width in physical units (minor and major axes)
            sigma_minor = width * min(Lx, Ly)
            sigma_major = sigma_minor * ar

            # Rotate coordinates
            X_rot = (X - cx) * np.cos(theta) + (Y - cy) * np.sin(theta)
            Y_rot = -(X - cx) * np.sin(theta) + (Y - cy) * np.cos(theta)

            # Add elliptical Gaussian blob
            blob = amplitude * np.exp(
                -(X_rot**2 / (2 * sigma_major**2) + Y_rot**2 / (2 * sigma_minor**2))
            )
            data += blob

        return ScalarField(grid, data)


class SingleBlob(InitialConditionGenerator):
    """Single Gaussian blob at a specified position.

    Creates a field with one Gaussian-shaped bump.
    """

    def generate(
        self,
        grid: CartesianGrid,
        amplitude: float = 1.0,
        width: float = 0.1,
        center_x: float = 0.5,
        center_y: float = 0.5,
        background: float = 0.0,
        **kwargs,
    ) -> ScalarField:
        """Generate single Gaussian blob initial condition.

        Args:
            grid: The computational grid.
            amplitude: Peak amplitude of the blob.
            width: Width of blob relative to domain size.
            center_x: X position of center (0-1 normalized).
            center_y: Y position of center (0-1 normalized).
            background: Background value.
            **kwargs: Additional arguments (ignored).

        Returns:
            ScalarField with single Gaussian blob.
        """
        # Get domain bounds
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[0]

        # Convert normalized position to physical coordinates
        cx = x_bounds[0] + center_x * Lx
        cy = y_bounds[0] + center_y * Ly

        # Create coordinate arrays
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Width in physical units
        sigma_x = width * Lx
        sigma_y = width * Ly

        # Create Gaussian blob
        data = background + amplitude * np.exp(
            -((X - cx) ** 2 / (2 * sigma_x**2) + (Y - cy) ** 2 / (2 * sigma_y**2))
        )

        return ScalarField(grid, data)

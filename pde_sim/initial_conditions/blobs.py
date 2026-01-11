"""Gaussian blob initial condition generators."""

import numpy as np
from pde import CartesianGrid, ScalarField

from .base import InitialConditionGenerator


class GaussianBlobs(InitialConditionGenerator):
    """Multiple Gaussian blobs initial condition.

    Creates a field with multiple Gaussian-shaped bumps at random positions.
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
            grid: The computational grid.
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

            # Blob amplitude
            if random_amplitude:
                amp = rng.uniform(0.5 * amplitude, amplitude)
            else:
                amp = amplitude

            # Width in physical units
            sigma_x = width * Lx
            sigma_y = width * Ly

            # Add Gaussian blob
            blob = amp * np.exp(
                -((X - cx) ** 2 / (2 * sigma_x**2) + (Y - cy) ** 2 / (2 * sigma_y**2))
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

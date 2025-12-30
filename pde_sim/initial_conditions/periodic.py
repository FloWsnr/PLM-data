"""Periodic/sinusoidal initial condition generators."""

import numpy as np
from pde import CartesianGrid, ScalarField

from .base import InitialConditionGenerator


class SinePattern(InitialConditionGenerator):
    """Sinusoidal pattern initial condition.

    Creates a field with sine waves in x and/or y directions.
    """

    def generate(
        self,
        grid: CartesianGrid,
        kx: int = 1,
        ky: int = 1,
        amplitude: float = 1.0,
        offset: float = 0.0,
        phase_x: float = 0.0,
        phase_y: float = 0.0,
        **kwargs,
    ) -> ScalarField:
        """Generate sinusoidal pattern initial condition.

        Args:
            grid: The computational grid.
            kx: Wavenumber in x direction (number of complete waves).
            ky: Wavenumber in y direction (number of complete waves).
            amplitude: Amplitude of the sine wave.
            offset: Constant offset (mean value).
            phase_x: Phase shift in x direction (in radians).
            phase_y: Phase shift in y direction (in radians).
            **kwargs: Additional arguments (ignored).

        Returns:
            ScalarField with sinusoidal pattern.
        """
        # Get domain bounds
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[0]

        # Create coordinate arrays
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Create sinusoidal pattern
        data = offset + amplitude * np.sin(
            2 * np.pi * kx * (X - x_bounds[0]) / Lx + phase_x
        ) * np.sin(2 * np.pi * ky * (Y - y_bounds[0]) / Ly + phase_y)

        return ScalarField(grid, data)


class CosinePattern(InitialConditionGenerator):
    """Cosine pattern initial condition.

    Creates a field with cosine waves - useful when you want the
    pattern to start at its maximum rather than zero.
    """

    def generate(
        self,
        grid: CartesianGrid,
        kx: int = 1,
        ky: int = 1,
        amplitude: float = 1.0,
        offset: float = 0.0,
        **kwargs,
    ) -> ScalarField:
        """Generate cosine pattern initial condition.

        Args:
            grid: The computational grid.
            kx: Wavenumber in x direction.
            ky: Wavenumber in y direction.
            amplitude: Amplitude of the cosine wave.
            offset: Constant offset (mean value).
            **kwargs: Additional arguments (ignored).

        Returns:
            ScalarField with cosine pattern.
        """
        # Get domain bounds
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[0]

        # Create coordinate arrays
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Create cosine pattern
        data = offset + amplitude * np.cos(
            2 * np.pi * kx * (X - x_bounds[0]) / Lx
        ) * np.cos(2 * np.pi * ky * (Y - y_bounds[0]) / Ly)

        return ScalarField(grid, data)

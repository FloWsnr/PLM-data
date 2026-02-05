"""Periodic/sinusoidal initial condition generators."""

import numpy as np
from pde import CartesianGrid, ScalarField

from .base import InitialConditionGenerator


class SinePattern(InitialConditionGenerator):
    """Sinusoidal pattern initial condition.

    Creates a field with sine waves. Supports 1D, 2D, and 3D grids.
    """

    def generate(
        self,
        grid: CartesianGrid,
        kx: int = 1,
        ky: int = 1,
        kz: int = 1,
        amplitude: float = 1.0,
        offset: float = 0.0,
        phase_x: float | None = None,
        phase_y: float | None = None,
        phase_z: float | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> ScalarField:
        """Generate sinusoidal pattern initial condition.

        Args:
            grid: The computational grid (1D, 2D, or 3D).
            kx: Wavenumber in x direction (number of complete waves).
            ky: Wavenumber in y direction (2D/3D only).
            kz: Wavenumber in z direction (3D only).
            amplitude: Amplitude of the sine wave.
            offset: Constant offset (mean value).
            phase_x: Phase shift in x direction (in radians).
            phase_y: Phase shift in y direction (in radians).
            phase_z: Phase shift in z direction (in radians).
            seed: Random seed for reproducibility (used during random resolution).
            **kwargs: Additional arguments (ignored).

        Returns:
            ScalarField with sinusoidal pattern.
        """
        ndim = len(grid.shape)
        if phase_x is None or phase_x == "random":
            raise ValueError("sine phase_x must be provided (use phase_x: random in config)")
        if ndim >= 2 and (phase_y is None or phase_y == "random"):
            raise ValueError("sine phase_y must be provided (use phase_y: random in config)")
        if ndim >= 3 and (phase_z is None or phase_z == "random"):
            raise ValueError("sine phase_z must be provided (use phase_z: random in config)")

        # Get domain bounds for x
        x_bounds = grid.axes_bounds[0]
        Lx = x_bounds[1] - x_bounds[0]
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])

        if ndim == 1:
            # 1D: sin(kx * x + phase)
            data = offset + amplitude * np.sin(
                2 * np.pi * kx * (x - x_bounds[0]) / Lx + phase_x
            )
        elif ndim == 2:
            # 2D: sin(kx * x) * sin(ky * y)
            y_bounds = grid.axes_bounds[1]
            Ly = y_bounds[1] - y_bounds[0]
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            data = offset + amplitude * np.sin(
                2 * np.pi * kx * (X - x_bounds[0]) / Lx + phase_x
            ) * np.sin(2 * np.pi * ky * (Y - y_bounds[0]) / Ly + phase_y)
        else:
            # 3D: sin(kx * x) * sin(ky * y) * sin(kz * z)
            y_bounds = grid.axes_bounds[1]
            z_bounds = grid.axes_bounds[2]
            Ly = y_bounds[1] - y_bounds[0]
            Lz = z_bounds[1] - z_bounds[0]
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            z = np.linspace(z_bounds[0], z_bounds[1], grid.shape[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

            data = offset + amplitude * np.sin(
                2 * np.pi * kx * (X - x_bounds[0]) / Lx + phase_x
            ) * np.sin(
                2 * np.pi * ky * (Y - y_bounds[0]) / Ly + phase_y
            ) * np.sin(
                2 * np.pi * kz * (Z - z_bounds[0]) / Lz + phase_z
            )

        return ScalarField(grid, data)

    @classmethod
    def get_position_params(cls) -> set[str]:
        """Return names of parameters that represent spatial phases."""
        return {"phase_x", "phase_y", "phase_z"}

    @classmethod
    def resolve_random_params(
        cls,
        grid: CartesianGrid,
        params: dict,
    ) -> dict:
        """Resolve random phases for sine pattern."""
        resolved = params.copy()
        ndim = len(grid.shape)
        phase_keys = ["phase_x"]
        if ndim >= 2:
            phase_keys.append("phase_y")
        if ndim >= 3:
            phase_keys.append("phase_z")

        for key in phase_keys:
            if key not in resolved:
                raise ValueError(f"sine requires {key} or {key}: random")

        seed = resolved.get("seed")
        rng = None
        for key in phase_keys:
            if resolved[key] == "random":
                if rng is None:
                    rng = np.random.default_rng(seed)
                resolved[key] = rng.uniform(0, 2 * np.pi)
            if resolved[key] is None:
                raise ValueError(f"sine requires {key} or {key}: random")

        return resolved


class CosinePattern(InitialConditionGenerator):
    """Cosine pattern initial condition.

    Creates a field with cosine waves - useful when you want the
    pattern to start at its maximum rather than zero.
    Supports 1D, 2D, and 3D grids.
    """

    def generate(
        self,
        grid: CartesianGrid,
        kx: int = 1,
        ky: int = 1,
        kz: int = 1,
        amplitude: float = 1.0,
        offset: float = 0.0,
        phase_x: float | None = None,
        phase_y: float | None = None,
        phase_z: float | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> ScalarField:
        """Generate cosine pattern initial condition.

        Args:
            grid: The computational grid (1D, 2D, or 3D).
            kx: Wavenumber in x direction.
            ky: Wavenumber in y direction (2D/3D only).
            kz: Wavenumber in z direction (3D only).
            amplitude: Amplitude of the cosine wave.
            offset: Constant offset (mean value).
            phase_x: Phase shift in x direction (in radians).
            phase_y: Phase shift in y direction (in radians).
            phase_z: Phase shift in z direction (in radians).
            seed: Random seed for reproducibility (used during random resolution).
            **kwargs: Additional arguments (ignored).

        Returns:
            ScalarField with cosine pattern.
        """
        ndim = len(grid.shape)
        if phase_x is None or phase_x == "random":
            raise ValueError("cosine phase_x must be provided (use phase_x: random in config)")
        if ndim >= 2 and (phase_y is None or phase_y == "random"):
            raise ValueError("cosine phase_y must be provided (use phase_y: random in config)")
        if ndim >= 3 and (phase_z is None or phase_z == "random"):
            raise ValueError("cosine phase_z must be provided (use phase_z: random in config)")

        # Get domain bounds for x
        x_bounds = grid.axes_bounds[0]
        Lx = x_bounds[1] - x_bounds[0]
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])

        if ndim == 1:
            # 1D: cos(kx * x + phase)
            data = offset + amplitude * np.cos(
                2 * np.pi * kx * (x - x_bounds[0]) / Lx + phase_x
            )
        elif ndim == 2:
            # 2D: cos(kx * x + phase_x) * cos(ky * y + phase_y)
            y_bounds = grid.axes_bounds[1]
            Ly = y_bounds[1] - y_bounds[0]
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            X, Y = np.meshgrid(x, y, indexing="ij")

            data = offset + amplitude * np.cos(
                2 * np.pi * kx * (X - x_bounds[0]) / Lx + phase_x
            ) * np.cos(2 * np.pi * ky * (Y - y_bounds[0]) / Ly + phase_y)
        else:
            # 3D: cos(kx * x + phase_x) * cos(ky * y + phase_y) * cos(kz * z + phase_z)
            y_bounds = grid.axes_bounds[1]
            z_bounds = grid.axes_bounds[2]
            Ly = y_bounds[1] - y_bounds[0]
            Lz = z_bounds[1] - z_bounds[0]
            y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
            z = np.linspace(z_bounds[0], z_bounds[1], grid.shape[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

            data = offset + amplitude * np.cos(
                2 * np.pi * kx * (X - x_bounds[0]) / Lx + phase_x
            ) * np.cos(
                2 * np.pi * ky * (Y - y_bounds[0]) / Ly + phase_y
            ) * np.cos(
                2 * np.pi * kz * (Z - z_bounds[0]) / Lz + phase_z
            )

        return ScalarField(grid, data)

    @classmethod
    def get_position_params(cls) -> set[str]:
        """Return names of parameters that represent spatial phases."""
        return {"phase_x", "phase_y", "phase_z"}

    @classmethod
    def resolve_random_params(
        cls,
        grid: CartesianGrid,
        params: dict,
    ) -> dict:
        """Resolve random phases for cosine pattern."""
        resolved = params.copy()
        ndim = len(grid.shape)
        phase_keys = ["phase_x"]
        if ndim >= 2:
            phase_keys.append("phase_y")
        if ndim >= 3:
            phase_keys.append("phase_z")

        for key in phase_keys:
            if key not in resolved:
                raise ValueError(f"cosine requires {key} or {key}: random")

        seed = resolved.get("seed")
        rng = None
        for key in phase_keys:
            if resolved[key] == "random":
                if rng is None:
                    rng = np.random.default_rng(seed)
                resolved[key] = rng.uniform(0, 2 * np.pi)
            if resolved[key] is None:
                raise ValueError(f"cosine requires {key} or {key}: random")

        return resolved

"""Step function and constant initial condition generators."""

import numpy as np
from pde import CartesianGrid, ScalarField

from .base import InitialConditionGenerator


class Constant(InitialConditionGenerator):
    """Constant (uniform) initial condition.

    Creates a field with a uniform value everywhere.
    Useful for plate equation with Dirichlet BC to create inward waves.
    """

    def generate(
        self,
        grid: CartesianGrid,
        value: float = 1.0,
        **kwargs,
    ) -> ScalarField:
        """Generate constant initial condition.

        Args:
            grid: The computational grid.
            value: The constant value everywhere.
            **kwargs: Additional arguments (ignored).

        Returns:
            ScalarField with uniform value.
        """
        data = np.full(grid.shape, value, dtype=float)
        return ScalarField(grid, data)


class StepFunction(InitialConditionGenerator):
    """Step function (Heaviside) initial condition.

    Creates a field with a sharp transition at a specified position.
    Supports 1D, 2D, and 3D grids.
    """

    def generate(
        self,
        grid: CartesianGrid,
        direction: str = "x",
        position: float | None = None,
        value_low: float = 0.0,
        value_high: float = 1.0,
        smooth_width: float = 0.0,
        seed: int | None = None,
        randomize: bool = False,
        **kwargs,
    ) -> ScalarField:
        """Generate step function initial condition.

        Args:
            grid: The computational grid (1D, 2D, or 3D).
            direction: Direction of the step ("x", "y", or "z").
            position: Position of the step (0-1 normalized).
            value_low: Value below/left of the step.
            value_high: Value above/right of the step.
            smooth_width: If > 0, use a smooth tanh transition.
            seed: Random seed for reproducibility.
            randomize: If True, randomize position.
            **kwargs: Additional arguments (ignored).

        Returns:
            ScalarField with step function.
        """
        if randomize:
            rng = np.random.default_rng(seed)
            position = rng.uniform(0.1, 0.9)

        if position is None:
            raise ValueError("step position must be provided")

        ndim = len(grid.shape)
        dir_map = {"x": 0, "y": 1, "z": 2}
        axis_idx = dir_map[direction.lower()]
        if axis_idx >= ndim:
            raise ValueError(f"direction '{direction}' invalid for {ndim}D grid")

        bounds = grid.axes_bounds[axis_idx]
        L = bounds[1] - bounds[0]
        coord_1d = np.linspace(bounds[0], bounds[1], grid.shape[axis_idx])

        # Reshape for broadcasting: e.g., y in 3D -> shape (1, ny, 1)
        shape = [1] * ndim
        shape[axis_idx] = grid.shape[axis_idx]
        coord = coord_1d.reshape(shape)

        step_pos = bounds[0] + position * L

        if smooth_width > 0:
            width = smooth_width * L
            transition = 0.5 * (1 + np.tanh((coord - step_pos) / width))
            data = value_low + (value_high - value_low) * transition
        else:
            data = np.where(coord < step_pos, value_low, value_high)

        # Broadcasting will expand to full grid shape
        data = np.broadcast_to(data, grid.shape).copy()

        return ScalarField(grid, data.astype(float))


class RectangleGrid(InitialConditionGenerator):
    """Grid of blocks with different values.

    Divides the domain into a grid of blocks (segments in 1D, rectangles
    in 2D, boxes in 3D), each assigned a different value (e.g., different
    pressures or concentrations). Supports 1D, 2D, and 3D grids.
    """

    def generate(
        self,
        grid: CartesianGrid,
        nx: int = 2,
        ny: int = 2,
        nz: int = 2,
        values: list | np.ndarray | None = None,
        value_range: tuple[float, float] = (0.0, 1.0),
        seed: int | None = None,
        **kwargs,
    ) -> ScalarField:
        """Generate a grid of blocks with different values.

        Args:
            grid: The computational grid (1D, 2D, or 3D).
            nx: Number of blocks in x direction.
            ny: Number of blocks in y direction (used for 2D/3D).
            nz: Number of blocks in z direction (used for 3D).
            values: Optional N-D array of values for each block.
                    Shape should match (nx,) for 1D, (nx, ny) for 2D,
                    or (nx, ny, nz) for 3D. If None, random values
                    are generated from value_range.
            value_range: (min, max) range for random value generation.
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments (ignored).

        Returns:
            ScalarField with block grid pattern.
        """
        ndim = len(grid.shape)
        divisions = [nx, ny, nz][:ndim]

        # Generate or validate values
        if values is None:
            rng = np.random.default_rng(seed)
            values = rng.uniform(value_range[0], value_range[1], size=tuple(divisions))
        else:
            values = np.array(values)
            if values.shape != tuple(divisions):
                raise ValueError(
                    f"values shape {values.shape} doesn't match grid {tuple(divisions)}"
                )

        # Build block indices per axis using digitize
        block_indices = []
        for i in range(ndim):
            bounds = grid.axes_bounds[i]
            coords = np.linspace(bounds[0], bounds[1], grid.shape[i])
            edges = np.linspace(bounds[0], bounds[1], divisions[i] + 1)
            idx = np.clip(np.digitize(coords, edges[1:]), 0, divisions[i] - 1)
            shape = [1] * ndim
            shape[i] = grid.shape[i]
            block_indices.append(idx.reshape(shape))

        # Use advanced indexing
        data = values[tuple(np.broadcast_arrays(*block_indices))]

        return ScalarField(grid, data.astype(float))


class DoubleStep(InitialConditionGenerator):
    """Double step function creating a band/stripe.

    Creates a field with a band of one value surrounded by another.
    Supports 1D, 2D, and 3D grids.
    """

    def generate(
        self,
        grid: CartesianGrid,
        direction: str = "x",
        position1: float | None = None,
        position2: float | None = None,
        value_inside: float = 1.0,
        value_outside: float = 0.0,
        smooth_width: float = 0.0,
        seed: int | None = None,
        randomize: bool = False,
        **kwargs,
    ) -> ScalarField:
        """Generate double step function (band) initial condition.

        Args:
            grid: The computational grid (1D, 2D, or 3D).
            direction: Direction of the bands ("x", "y", or "z").
            position1: Position of first step (0-1 normalized).
            position2: Position of second step (0-1 normalized).
            value_inside: Value inside the band.
            value_outside: Value outside the band.
            smooth_width: If > 0, use smooth tanh transitions.
            seed: Random seed for reproducibility.
            randomize: If True, randomize positions.
            **kwargs: Additional arguments (ignored).

        Returns:
            ScalarField with double step function.
        """
        if randomize:
            rng = np.random.default_rng(seed)
            position1 = rng.uniform(0.1, 0.4)
            position2 = rng.uniform(0.6, 0.9)
            if position1 >= position2:
                position1, position2 = position2, position1
            if position2 - position1 < 0.1:
                position2 = position1 + 0.1

        if position1 is None or position2 is None:
            raise ValueError("double-step requires position1 and position2")

        ndim = len(grid.shape)
        dir_map = {"x": 0, "y": 1, "z": 2}
        axis_idx = dir_map[direction.lower()]
        if axis_idx >= ndim:
            raise ValueError(f"direction '{direction}' invalid for {ndim}D grid")

        bounds = grid.axes_bounds[axis_idx]
        L = bounds[1] - bounds[0]
        coord_1d = np.linspace(bounds[0], bounds[1], grid.shape[axis_idx])

        shape = [1] * ndim
        shape[axis_idx] = grid.shape[axis_idx]
        coord = coord_1d.reshape(shape)

        pos1 = bounds[0] + position1 * L
        pos2 = bounds[0] + position2 * L

        if smooth_width > 0:
            width = smooth_width * L
            trans1 = 0.5 * (1 + np.tanh((coord - pos1) / width))
            trans2 = 0.5 * (1 + np.tanh((pos2 - coord) / width))
            inside = trans1 * trans2
            data = value_outside + (value_inside - value_outside) * inside
        else:
            inside = (coord >= pos1) & (coord <= pos2)
            data = np.where(inside, value_inside, value_outside)

        data = np.broadcast_to(data, grid.shape).copy()
        return ScalarField(grid, data.astype(float))

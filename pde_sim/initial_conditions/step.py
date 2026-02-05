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
        **kwargs,
    ) -> ScalarField:
        """Generate step function initial condition.

        Args:
            grid: The computational grid.
            direction: Direction of the step ("x" or "y").
            position: Position of the step (0-1 normalized). If None, randomized.
            value_low: Value below/left of the step.
            value_high: Value above/right of the step.
            smooth_width: If > 0, use a smooth tanh transition.
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments (ignored).

        Returns:
            ScalarField with step function.
        """
        if position is None:
            rng = np.random.default_rng(seed)
            position = rng.uniform(0.1, 0.9)
        # Get domain bounds
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[0]

        # Create coordinate arrays
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        if direction.lower() == "x":
            step_pos = x_bounds[0] + position * Lx
            coord = X
            L = Lx
        else:
            step_pos = y_bounds[0] + position * Ly
            coord = Y
            L = Ly

        if smooth_width > 0:
            # Smooth transition using tanh
            width = smooth_width * L
            transition = 0.5 * (1 + np.tanh((coord - step_pos) / width))
            data = value_low + (value_high - value_low) * transition
        else:
            # Sharp step
            data = np.where(coord < step_pos, value_low, value_high)

        return ScalarField(grid, data.astype(float))


class RectangleGrid(InitialConditionGenerator):
    """Grid of rectangles with different values.

    Divides the domain into a grid of rectangles, each assigned
    a different value (e.g., different pressures or concentrations).
    """

    def generate(
        self,
        grid: CartesianGrid,
        nx: int = 2,
        ny: int = 2,
        values: list[list[float]] | None = None,
        value_range: tuple[float, float] = (0.0, 1.0),
        seed: int | None = None,
        **kwargs,
    ) -> ScalarField:
        """Generate a grid of rectangles with different values.

        Args:
            grid: The computational grid.
            nx: Number of rectangles in x direction.
            ny: Number of rectangles in y direction.
            values: Optional 2D list of values for each rectangle.
                    Shape should be (nx, ny). If None, random values
                    are generated from value_range.
            value_range: (min, max) range for random value generation.
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments (ignored).

        Returns:
            ScalarField with rectangle grid pattern.
        """
        # Get domain bounds
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]

        # Create coordinate arrays
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Generate or validate values
        if values is None:
            rng = np.random.default_rng(seed)
            values = rng.uniform(value_range[0], value_range[1], size=(nx, ny))
        else:
            values = np.array(values)
            if values.shape != (nx, ny):
                raise ValueError(
                    f"values shape {values.shape} doesn't match grid ({nx}, {ny})"
                )

        # Calculate rectangle boundaries
        x_edges = np.linspace(x_bounds[0], x_bounds[1], nx + 1)
        y_edges = np.linspace(y_bounds[0], y_bounds[1], ny + 1)

        # Assign values to each grid cell
        data = np.zeros(grid.shape)
        for i in range(nx):
            for j in range(ny):
                mask = (
                    (X >= x_edges[i])
                    & (X < x_edges[i + 1])
                    & (Y >= y_edges[j])
                    & (Y < y_edges[j + 1])
                )
                data[mask] = values[i, j]

        # Handle edge case: rightmost/topmost cells (X == x_max or Y == y_max)
        data[X == x_bounds[1]] = values[-1, :][
            np.searchsorted(y_edges[:-1], Y[X == x_bounds[1]], side="right") - 1
        ]
        data[Y == y_bounds[1]] = values[:, -1][
            np.searchsorted(x_edges[:-1], X[Y == y_bounds[1]], side="right") - 1
        ]

        return ScalarField(grid, data.astype(float))


class DoubleStep(InitialConditionGenerator):
    """Double step function creating a band/stripe.

    Creates a field with a band of one value surrounded by another.
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
        **kwargs,
    ) -> ScalarField:
        """Generate double step function (band) initial condition.

        Args:
            grid: The computational grid.
            direction: Direction of the bands ("x" or "y").
            position1: Position of first step (0-1 normalized). If None, randomized.
            position2: Position of second step (0-1 normalized). If None, randomized.
            value_inside: Value inside the band.
            value_outside: Value outside the band.
            smooth_width: If > 0, use smooth tanh transitions.
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments (ignored).

        Returns:
            ScalarField with double step function.
        """
        if position1 is None or position2 is None:
            rng = np.random.default_rng(seed)
            if position1 is None:
                position1 = rng.uniform(0.1, 0.4)
            if position2 is None:
                position2 = rng.uniform(0.6, 0.9)
            if position1 >= position2:
                position1, position2 = position2, position1
            # Ensure minimum gap
            if position2 - position1 < 0.1:
                position2 = position1 + 0.1
        # Get domain bounds
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[0]

        # Create coordinate arrays
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        if direction.lower() == "x":
            pos1 = x_bounds[0] + position1 * Lx
            pos2 = x_bounds[0] + position2 * Lx
            coord = X
            L = Lx
        else:
            pos1 = y_bounds[0] + position1 * Ly
            pos2 = y_bounds[0] + position2 * Ly
            coord = Y
            L = Ly

        if smooth_width > 0:
            width = smooth_width * L
            trans1 = 0.5 * (1 + np.tanh((coord - pos1) / width))
            trans2 = 0.5 * (1 + np.tanh((pos2 - coord) / width))
            inside = trans1 * trans2
            data = value_outside + (value_inside - value_outside) * inside
        else:
            inside = (coord >= pos1) & (coord <= pos2)
            data = np.where(inside, value_inside, value_outside)

        return ScalarField(grid, data.astype(float))

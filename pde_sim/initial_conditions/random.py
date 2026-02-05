"""Random initial condition generators."""

import numpy as np
from pde import CartesianGrid, ScalarField

from .base import InitialConditionGenerator


class RandomUniform(InitialConditionGenerator):
    """Uniform random initial condition.

    Generates values uniformly distributed between low and high.
    """

    def generate(
        self,
        grid: CartesianGrid,
        low: float = 0.0,
        high: float = 1.0,
        seed: int | None = None,
        **kwargs,
    ) -> ScalarField:
        """Generate uniform random initial condition.

        Args:
            grid: The computational grid.
            low: Lower bound of the uniform distribution.
            high: Upper bound of the uniform distribution.
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments (ignored).

        Returns:
            ScalarField with uniform random values.
        """
        rng = np.random.default_rng(seed)
        data = rng.uniform(low, high, grid.shape)
        return ScalarField(grid, data)


class RandomGaussian(InitialConditionGenerator):
    """Gaussian (normal) random initial condition.

    Generates values from a normal distribution with specified mean and std.
    """

    def generate(
        self,
        grid: CartesianGrid,
        mean: float = 0.0,
        std: float = 1.0,
        clip_min: float | None = None,
        clip_max: float | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> ScalarField:
        """Generate Gaussian random initial condition.

        Args:
            grid: The computational grid.
            mean: Mean of the normal distribution.
            std: Standard deviation of the normal distribution.
            clip_min: Optional minimum value to clip to.
            clip_max: Optional maximum value to clip to.
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments (ignored).

        Returns:
            ScalarField with Gaussian random values.
        """
        rng = np.random.default_rng(seed)
        data = rng.normal(mean, std, grid.shape)

        if clip_min is not None or clip_max is not None:
            data = np.clip(data, clip_min, clip_max)

        return ScalarField(grid, data)

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

"""Abstract base class for initial condition generators."""

from abc import ABC, abstractmethod

from pde import CartesianGrid, ScalarField


class InitialConditionGenerator(ABC):
    """Abstract base class for initial condition generators.

    Subclasses must implement the generate method to create
    a ScalarField with the desired initial values.
    """

    @abstractmethod
    def generate(self, grid: CartesianGrid, **params) -> ScalarField:
        """Generate an initial condition field.

        Args:
            grid: The computational grid.
            **params: Generator-specific parameters.

        Returns:
            A ScalarField with the initial values.
        """
        pass

    @classmethod
    def resolve_random_params(
        cls,
        grid: CartesianGrid,
        params: dict,
    ) -> dict:
        """Resolve any "random" placeholders in parameters.

        Args:
            grid: The computational grid.
            params: Generator-specific parameters.

        Returns:
            Parameters with any random placeholders resolved.
        """
        return params

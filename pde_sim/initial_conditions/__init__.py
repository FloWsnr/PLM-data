"""Initial condition generators."""

from typing import Any

from pde import CartesianGrid, ScalarField

from .base import InitialConditionGenerator
from .blobs import GaussianBlobs
from .periodic import SinePattern
from .random import RandomGaussian, RandomUniform
from .step import StepFunction

# Registry of available initial condition generators
_IC_REGISTRY: dict[str, type[InitialConditionGenerator]] = {
    "random-uniform": RandomUniform,
    "random-gaussian": RandomGaussian,
    "gaussian-blobs": GaussianBlobs,
    "sine": SinePattern,
    "step": StepFunction,
}


def create_initial_condition(
    grid: CartesianGrid,
    ic_type: str,
    params: dict[str, Any],
) -> ScalarField:
    """Factory function to create initial conditions.

    Args:
        grid: The computational grid.
        ic_type: Type of initial condition to generate.
        params: Parameters for the initial condition generator.

    Returns:
        Generated scalar field.

    Raises:
        ValueError: If ic_type is not recognized.
    """
    if ic_type not in _IC_REGISTRY:
        available = ", ".join(sorted(_IC_REGISTRY.keys()))
        raise ValueError(f"Unknown IC type: {ic_type}. Available: {available}")

    generator = _IC_REGISTRY[ic_type]()
    return generator.generate(grid, **params)


def list_initial_conditions() -> list[str]:
    """List all available initial condition types.

    Returns:
        Sorted list of registered IC type names.
    """
    return sorted(_IC_REGISTRY.keys())


def register_initial_condition(name: str):
    """Decorator to register a new initial condition generator.

    Args:
        name: The name to register the generator under.

    Returns:
        Decorator function that registers the class.
    """

    def decorator(cls: type[InitialConditionGenerator]) -> type[InitialConditionGenerator]:
        _IC_REGISTRY[name] = cls
        return cls

    return decorator


__all__ = [
    "InitialConditionGenerator",
    "RandomUniform",
    "RandomGaussian",
    "GaussianBlobs",
    "SinePattern",
    "StepFunction",
    "create_initial_condition",
    "list_initial_conditions",
    "register_initial_condition",
]

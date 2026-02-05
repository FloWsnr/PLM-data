"""Initial condition generators."""

from typing import Any

from pde import CartesianGrid, ScalarField

from .base import InitialConditionGenerator
from .blobs import GaussianBlob
from .periodic import CosinePattern, SinePattern
from .random import RandomGaussian, RandomUniform
from .step import Constant, DoubleStep, RectangleGrid, StepFunction

# Registry of available initial condition generators
_IC_REGISTRY: dict[str, type[InitialConditionGenerator]] = {
    "random-uniform": RandomUniform,
    "random-gaussian": RandomGaussian,
    "gaussian-blob": GaussianBlob,
    "sine": SinePattern,
    "cosine": CosinePattern,
    "step": StepFunction,
    "double-step": DoubleStep,
    "rectangle-grid": RectangleGrid,
    "constant": Constant,
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


def resolve_random_params(
    grid: CartesianGrid,
    ic_type: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Resolve random placeholders for an initial condition type.

    Args:
        grid: The computational grid.
        ic_type: Type of initial condition to resolve.
        params: Parameters for the initial condition generator.

    Returns:
        Parameters with any random placeholders resolved.
    """
    if ic_type not in _IC_REGISTRY:
        available = ", ".join(sorted(_IC_REGISTRY.keys()))
        raise ValueError(f"Unknown IC type: {ic_type}. Available: {available}")

    generator_cls = _IC_REGISTRY[ic_type]
    return generator_cls.resolve_random_params(grid, params.copy())


def resolve_initial_condition_params(
    grid: CartesianGrid,
    ic_type: str,
    ic_params: dict[str, Any],
    field_names: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve random placeholders for IC params, including per-field overrides.

    Args:
        grid: The computational grid.
        ic_type: Default initial condition type.
        ic_params: Initial condition parameters (may include per-field overrides).
        field_names: Optional list of field names for per-field overrides.

    Returns:
        Parameters with any random placeholders resolved.
    """
    if field_names is None:
        return resolve_random_params(grid, ic_type, ic_params)

    resolved: dict[str, Any] = {}
    for name in field_names:
        if name in ic_params and isinstance(ic_params[name], dict):
            field_spec = ic_params[name]
            if "type" in field_spec:
                field_type = field_spec["type"]
            else:
                field_type = ic_type
            if "params" in field_spec:
                field_params = field_spec["params"]
            else:
                field_params = {}
            resolved_params = resolve_random_params(grid, field_type, field_params)
            resolved[name] = {"type": field_type, "params": resolved_params}

    for key, value in ic_params.items():
        if field_names is not None and key in field_names:
            if key in resolved:
                continue
        resolved[key] = value

    non_field_params: dict[str, Any] = {
        key: value for key, value in resolved.items() if key not in field_names
    }
    resolved_non_fields = resolve_random_params(grid, ic_type, non_field_params)
    for key, value in resolved_non_fields.items():
        resolved[key] = value

    return resolved


def get_ic_position_params(ic_type: str) -> set[str]:
    """Return position parameter names for an initial condition type.

    Args:
        ic_type: Type of initial condition.

    Returns:
        Set of parameter names that represent spatial positions/phases.

    Raises:
        ValueError: If ic_type is not recognized.
    """
    if ic_type not in _IC_REGISTRY:
        available = ", ".join(sorted(_IC_REGISTRY.keys()))
        raise ValueError(f"Unknown IC type: {ic_type}. Available: {available}")

    return _IC_REGISTRY[ic_type].get_position_params()


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
    "GaussianBlob",
    "CosinePattern",
    "SinePattern",
    "DoubleStep",
    "StepFunction",
    "RectangleGrid",
    "Constant",
    "create_initial_condition",
    "resolve_random_params",
    "resolve_initial_condition_params",
    "get_ic_position_params",
    "list_initial_conditions",
    "register_initial_condition",
]

"""PDE preset registry and factory functions."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import PDEPreset

# Registry of all available PDE presets
_REGISTRY: dict[str, type["PDEPreset"]] = {}


def register_pde(name: str):
    """Decorator to register a PDE preset.

    Args:
        name: The name to register the preset under.

    Returns:
        Decorator function that registers the class.

    Example:
        @register_pde("heat")
        class HeatPDE(ScalarPDEPreset):
            ...
    """

    def decorator(cls: type["PDEPreset"]) -> type["PDEPreset"]:
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_pde_preset(name: str) -> "PDEPreset":
    """Get a PDE preset instance by name.

    Args:
        name: The registered name of the preset.

    Returns:
        An instance of the requested PDE preset.

    Raises:
        ValueError: If the preset name is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown PDE preset: {name}. Available: {available}")
    return _REGISTRY[name]()


def list_presets() -> list[str]:
    """List all available preset names.

    Returns:
        Sorted list of registered preset names.
    """
    return sorted(_REGISTRY.keys())


def get_presets_by_category() -> dict[str, list[str]]:
    """Get presets organized by category.

    Returns:
        Dictionary mapping category names to lists of preset names.
    """
    categories: dict[str, list[str]] = {}
    for name in _REGISTRY:
        preset = _REGISTRY[name]()
        category = preset.metadata.category
        if category not in categories:
            categories[category] = []
        categories[category].append(name)

    # Sort presets within each category
    for category in categories:
        categories[category].sort()

    return categories


# Import all PDE modules to trigger registration
# These imports must be at the bottom to avoid circular imports
from . import basic  # noqa: E402, F401
from . import biology  # noqa: E402, F401
from . import fluids  # noqa: E402, F401
from . import physics  # noqa: E402, F401

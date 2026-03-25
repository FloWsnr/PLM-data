"""Preset registry and module discovery."""

from __future__ import annotations

import importlib
import pkgutil

from plm_data.presets.base import PDEPreset

_REGISTRY: dict[str, type[PDEPreset]] = {}


def register_preset(name: str):
    """Decorator to register a PDE preset class."""

    def decorator(cls: type[PDEPreset]) -> type[PDEPreset]:
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_preset(name: str) -> PDEPreset:
    """Instantiate a registered preset by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return _REGISTRY[name]()


def list_presets() -> dict[str, type[PDEPreset]]:
    """Return all registered presets."""
    return dict(_REGISTRY)


def _load_all_presets() -> None:
    """Recursively import all preset modules under this package."""
    prefix = __name__ + "."
    skip_modules = {
        __name__ + ".base",
        __name__ + ".metadata",
    }
    for module_info in pkgutil.walk_packages(__path__, prefix):
        if module_info.name in skip_modules:
            continue
        importlib.import_module(module_info.name)


_load_all_presets()

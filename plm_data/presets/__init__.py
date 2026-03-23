"""PDE preset registry."""

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


def _load_all_presets():
    """Import all preset modules so they register themselves."""
    import plm_data.presets.basic  # noqa: F401
    import plm_data.presets.fluids  # noqa: F401
    import plm_data.presets.physics  # noqa: F401


_load_all_presets()

"""First-class initial-condition specs and discovery."""

import importlib
import pkgutil

from plm_data.initial_conditions.base import (
    COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
    InitialConditionParameterSpec,
    InitialConditionSpec,
    get_initial_condition_spec,
    has_initial_condition_spec,
    list_initial_condition_specs,
    register_initial_condition_spec,
)


def _load_package_modules(package_name: str) -> None:
    """Import all modules under one initial-condition spec package."""
    package = importlib.import_module(package_name)
    prefix = package.__name__ + "."
    for module_info in pkgutil.walk_packages(package.__path__, prefix):
        importlib.import_module(module_info.name)


_load_package_modules(__name__ + ".families")

__all__ = [
    "COMMON_SCALAR_INITIAL_CONDITION_FAMILIES",
    "InitialConditionParameterSpec",
    "InitialConditionSpec",
    "get_initial_condition_spec",
    "has_initial_condition_spec",
    "list_initial_condition_specs",
    "register_initial_condition_spec",
]

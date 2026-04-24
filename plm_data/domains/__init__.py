"""First-class domain registry and domain metadata discovery."""

import importlib
import pkgutil
from typing import Any

_DOMAIN_PACKAGE_SKIP_MODULES = {"base", "gmsh", "helpers", "validation"}
_DOMAIN_MODULES_LOADED = False
_DOMAIN_FACTORY_MODULES_LOADED = False

_BASE_EXPORTS = {
    "COMMON_BOUNDARY_FAMILIES",
    "COMMON_SCALAR_INITIAL_CONDITION_FAMILIES",
    "DomainGeometry",
    "DomainParameterSpec",
    "DomainSpec",
    "PeriodicBoundaryMap",
    "build_gmsh_domain_model",
    "build_gmsh_planar_domain_model",
    "create_domain",
    "get_domain_spec",
    "get_gmsh_domain_dimension",
    "is_gmsh_domain",
    "is_gmsh_planar_domain",
    "list_domain_specs",
    "list_domains",
    "register_domain",
    "register_domain_spec",
    "register_gmsh_domain",
    "register_gmsh_planar_domain",
}
_VALIDATION_EXPORTS = {
    "DomainConfigLike",
    "infer_domain_dimension",
    "validate_domain_params",
}

__all__ = [
    "COMMON_BOUNDARY_FAMILIES",
    "COMMON_SCALAR_INITIAL_CONDITION_FAMILIES",
    "DomainConfigLike",
    "DomainGeometry",
    "DomainParameterSpec",
    "DomainSpec",
    "PeriodicBoundaryMap",
    "build_gmsh_domain_model",
    "build_gmsh_planar_domain_model",
    "create_domain",
    "get_domain_spec",
    "get_gmsh_domain_dimension",
    "infer_domain_dimension",
    "is_gmsh_domain",
    "is_gmsh_planar_domain",
    "list_domain_specs",
    "list_domains",
    "register_domain",
    "register_domain_spec",
    "register_gmsh_domain",
    "register_gmsh_planar_domain",
    "validate_domain_params",
]


def _iter_domain_package_names() -> list[str]:
    return [
        module_info.name
        for module_info in pkgutil.iter_modules(__path__)
        if module_info.ispkg and module_info.name not in _DOMAIN_PACKAGE_SKIP_MODULES
    ]


def _load_domain_spec_modules() -> None:
    """Import domain specs without importing Gmsh builders."""
    global _DOMAIN_MODULES_LOADED
    if _DOMAIN_MODULES_LOADED:
        return
    _DOMAIN_MODULES_LOADED = True
    for package_name in _iter_domain_package_names():
        importlib.import_module(f"{__name__}.{package_name}.spec")


def _load_domain_factory_modules() -> None:
    """Import domain factories and Gmsh builders for mesh creation."""
    global _DOMAIN_FACTORY_MODULES_LOADED
    if _DOMAIN_FACTORY_MODULES_LOADED:
        return
    _DOMAIN_FACTORY_MODULES_LOADED = True
    _load_domain_spec_modules()
    for package_name in _iter_domain_package_names():
        importlib.import_module(f"{__name__}.{package_name}.gmsh")


def __getattr__(name: str) -> Any:
    if name in _VALIDATION_EXPORTS:
        validation = importlib.import_module(f"{__name__}.validation")
        value = getattr(validation, name)
        globals()[name] = value
        return value

    if name in _BASE_EXPORTS:
        _load_domain_spec_modules()
        base = importlib.import_module(f"{__name__}.base")
        value = getattr(base, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

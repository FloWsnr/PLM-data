"""First-class domain registry and domain metadata discovery."""

import importlib
import pkgutil

from plm_data.domains.base import (
    COMMON_BOUNDARY_FAMILIES,
    COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
    DomainGeometry,
    DomainParameterSpec,
    DomainSpec,
    PeriodicBoundaryMap,
    build_gmsh_domain_model,
    build_gmsh_planar_domain_model,
    create_domain,
    get_domain_spec,
    get_gmsh_domain_dimension,
    is_gmsh_domain,
    is_gmsh_planar_domain,
    list_domain_specs,
    list_domains,
    register_domain,
    register_domain_spec,
    register_gmsh_domain,
    register_gmsh_planar_domain,
)

__all__ = [
    "DomainGeometry",
    "DomainParameterSpec",
    "DomainSpec",
    "COMMON_BOUNDARY_FAMILIES",
    "COMMON_SCALAR_INITIAL_CONDITION_FAMILIES",
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
]


def _load_all_domain_modules() -> None:
    """Recursively import domain spec/factory modules under this package."""
    prefix = __name__ + "."
    skip_modules = {
        __name__ + ".base",
        __name__ + ".helpers",
    }
    for module_info in pkgutil.walk_packages(__path__, prefix):
        if module_info.name in skip_modules:
            continue
        importlib.import_module(module_info.name)


_load_all_domain_modules()

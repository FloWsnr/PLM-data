"""First-class boundary-condition specs and discovery."""

import importlib
import pkgutil

from plm_data.boundary_conditions.base import (
    BoundaryOperatorParameterSpec,
    BoundaryOperatorSpec,
    get_boundary_operator_spec,
    list_boundary_operator_specs,
    register_boundary_operator_spec,
)


def _load_package_modules(package_name: str) -> None:
    """Import all modules under one boundary-condition spec package."""
    package = importlib.import_module(package_name)
    prefix = package.__name__ + "."
    for module_info in pkgutil.walk_packages(package.__path__, prefix):
        importlib.import_module(module_info.name)


_load_package_modules(__name__ + ".operators")

SCALAR_STANDARD_BOUNDARY_OPERATORS = {
    name: get_boundary_operator_spec(name)
    for name in ("dirichlet", "neumann", "robin", "periodic")
}
VECTOR_STANDARD_BOUNDARY_OPERATORS = {
    name: get_boundary_operator_spec(name)
    for name in ("dirichlet", "neumann", "periodic")
}
MAXWELL_BOUNDARY_OPERATORS = {
    name: get_boundary_operator_spec(name)
    for name in ("dirichlet", "periodic", "absorbing")
}

__all__ = [
    "BoundaryOperatorParameterSpec",
    "BoundaryOperatorSpec",
    "MAXWELL_BOUNDARY_OPERATORS",
    "SCALAR_STANDARD_BOUNDARY_OPERATORS",
    "VECTOR_STANDARD_BOUNDARY_OPERATORS",
    "get_boundary_operator_spec",
    "list_boundary_operator_specs",
    "register_boundary_operator_spec",
]

"""Boundary-condition scenario discovery."""

import importlib
import pkgutil

from plm_data.boundary_conditions.scenarios.base import (
    BoundaryScenario,
    BoundaryScenarioSpec,
    compatible_boundary_scenarios,
    get_boundary_scenario,
    list_boundary_scenarios,
    register_boundary_scenario,
)


def _load_scenario_modules() -> None:
    prefix = __name__ + "."
    skip_modules = {__name__ + ".base"}
    for module_info in pkgutil.walk_packages(__path__, prefix):
        if module_info.name in skip_modules:
            continue
        importlib.import_module(module_info.name)


_load_scenario_modules()

__all__ = [
    "BoundaryScenario",
    "BoundaryScenarioSpec",
    "compatible_boundary_scenarios",
    "get_boundary_scenario",
    "list_boundary_scenarios",
    "register_boundary_scenario",
]

"""Initial-condition scenario discovery."""

import importlib
import pkgutil

from plm_data.initial_conditions.scenarios.base import (
    InitialConditionScenario,
    InitialConditionScenarioSpec,
    compatible_initial_condition_scenarios,
    get_initial_condition_scenario,
    list_initial_condition_scenarios,
    register_initial_condition_scenario,
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
    "InitialConditionScenario",
    "InitialConditionScenarioSpec",
    "compatible_initial_condition_scenarios",
    "get_initial_condition_scenario",
    "list_initial_condition_scenarios",
    "register_initial_condition_scenario",
]

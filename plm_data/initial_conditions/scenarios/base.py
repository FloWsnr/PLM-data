"""Initial-condition scenario specs and registry."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from plm_data.core.runtime_config import DomainConfig, InputConfig

if TYPE_CHECKING:
    from plm_data.sampling.context import SamplingContext

InitialConditionScenarioBuilder = Callable[
    ["SamplingContext", DomainConfig, dict[str, float]],
    dict[str, InputConfig],
]


@dataclass(frozen=True)
class InitialConditionScenarioSpec:
    """Compatibility metadata for one PDE-level initial-condition scenario."""

    name: str
    description: str
    supported_dimensions: tuple[int, ...]
    supported_pdes: tuple[str, ...]
    supported_domains: tuple[str, ...]
    configured_inputs: tuple[str, ...]
    field_shapes: tuple[str, ...]
    operators: tuple[str, ...]
    coordinate_regions: tuple[str, ...]

    def is_compatible(self, *, pde_name: str, domain_name: str) -> bool:
        """Return whether this scenario supports one PDE/domain pair."""
        if (
            pde_name not in self.supported_pdes
            or domain_name not in self.supported_domains
        ):
            return False

        from plm_data.domains.registry import get_domain_spec

        try:
            domain_spec = get_domain_spec(domain_name)
        except ValueError:
            return False

        if (
            domain_spec.supported_initial_condition_scenarios
            and self.name not in domain_spec.supported_initial_condition_scenarios
        ):
            return False
        return domain_spec.dimension in self.supported_dimensions


@dataclass(frozen=True)
class InitialConditionScenario:
    """Executable initial-condition scenario."""

    spec: InitialConditionScenarioSpec
    build: InitialConditionScenarioBuilder


_INITIAL_CONDITION_SCENARIOS: dict[str, InitialConditionScenario] = {}


def register_initial_condition_scenario(
    scenario: InitialConditionScenario,
) -> InitialConditionScenario:
    """Register one initial-condition scenario."""
    _INITIAL_CONDITION_SCENARIOS[scenario.spec.name] = scenario
    return scenario


def list_initial_condition_scenarios() -> dict[str, InitialConditionScenario]:
    """Return all registered initial-condition scenarios."""
    return dict(_INITIAL_CONDITION_SCENARIOS)


def compatible_initial_condition_scenarios(
    *,
    pde_name: str,
    domain_name: str,
) -> dict[str, InitialConditionScenario]:
    """Return IC scenarios compatible with a PDE/domain pair."""
    return {
        name: scenario
        for name, scenario in _INITIAL_CONDITION_SCENARIOS.items()
        if scenario.spec.is_compatible(pde_name=pde_name, domain_name=domain_name)
    }


def get_initial_condition_scenario(name: str) -> InitialConditionScenario:
    """Return one registered initial-condition scenario by name."""
    if name not in _INITIAL_CONDITION_SCENARIOS:
        available = ", ".join(sorted(_INITIAL_CONDITION_SCENARIOS))
        raise ValueError(
            f"Unknown initial-condition scenario '{name}'. Available: {available}"
        )
    return _INITIAL_CONDITION_SCENARIOS[name]

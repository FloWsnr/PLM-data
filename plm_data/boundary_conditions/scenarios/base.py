"""Boundary-condition scenario specs and registry."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from plm_data.core.runtime_config import BoundaryFieldConfig, DomainConfig

if TYPE_CHECKING:
    from plm_data.sampling.context import SamplingContext

BoundaryScenarioBuilder = Callable[
    ["SamplingContext", DomainConfig],
    dict[str, BoundaryFieldConfig],
]


@dataclass(frozen=True)
class BoundaryScenarioSpec:
    """Compatibility metadata for one complete PDE-level boundary scenario."""

    name: str
    description: str
    supported_dimensions: tuple[int, ...]
    supported_pdes: tuple[str, ...]
    supported_domains: tuple[str, ...]
    required_boundary_roles: tuple[str, ...]
    required_boundary_names: tuple[str, ...]
    configured_fields: tuple[str, ...]
    field_shapes: tuple[str, ...]
    operators: tuple[str, ...]
    level: str

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

        if domain_spec.dimension not in self.supported_dimensions:
            return False
        if (
            domain_spec.supported_boundary_scenarios
            and self.name not in domain_spec.supported_boundary_scenarios
        ):
            return False
        missing_roles = set(self.required_boundary_roles) - set(
            domain_spec.boundary_roles
        )
        if missing_roles:
            return False
        missing_boundaries = set(self.required_boundary_names) - set(
            domain_spec.boundary_names
        )
        return not missing_boundaries


@dataclass(frozen=True)
class BoundaryScenario:
    """Executable boundary scenario."""

    spec: BoundaryScenarioSpec
    build: BoundaryScenarioBuilder


_BOUNDARY_SCENARIOS: dict[str, BoundaryScenario] = {}


def register_boundary_scenario(scenario: BoundaryScenario) -> BoundaryScenario:
    """Register one boundary scenario."""
    _BOUNDARY_SCENARIOS[scenario.spec.name] = scenario
    return scenario


def list_boundary_scenarios() -> dict[str, BoundaryScenario]:
    """Return all registered boundary scenarios."""
    return dict(_BOUNDARY_SCENARIOS)


def compatible_boundary_scenarios(
    *,
    pde_name: str,
    domain_name: str,
) -> dict[str, BoundaryScenario]:
    """Return boundary scenarios compatible with a PDE/domain pair."""
    return {
        name: scenario
        for name, scenario in _BOUNDARY_SCENARIOS.items()
        if scenario.spec.is_compatible(pde_name=pde_name, domain_name=domain_name)
    }


def get_boundary_scenario(name: str) -> BoundaryScenario:
    """Return one registered boundary scenario by name."""
    if name not in _BOUNDARY_SCENARIOS:
        available = ", ".join(sorted(_BOUNDARY_SCENARIOS))
        raise ValueError(f"Unknown boundary scenario '{name}'. Available: {available}")
    return _BOUNDARY_SCENARIOS[name]

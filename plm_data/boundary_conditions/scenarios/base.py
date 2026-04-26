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


def _supports(value: str, supported_values: tuple[str, ...]) -> bool:
    return "*" in supported_values or value in supported_values


def _scenario_field_names(
    configured_fields: tuple[str, ...],
    pde_fields,
) -> tuple[str, ...] | None:
    if configured_fields == ("all_scalar_boundary_fields",):
        if any(field.shape != "scalar" for field in pde_fields.values()):
            return None
        return tuple(pde_fields)
    missing = set(configured_fields) - set(pde_fields)
    if missing:
        return None
    return configured_fields


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
        if not _supports(pde_name, self.supported_pdes) or not _supports(
            domain_name,
            self.supported_domains,
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
        if missing_boundaries:
            return False

        if "periodic" in self.operators:
            paired_boundaries = {
                side for pair in domain_spec.periodic_pairs for side in pair
            }
            if set(domain_spec.boundary_names) - paired_boundaries:
                return False

        from plm_data.pdes import get_pde

        try:
            pde_spec = get_pde(pde_name).spec
        except ValueError:
            return False
        field_names = _scenario_field_names(
            self.configured_fields,
            pde_spec.boundary_fields,
        )
        if field_names is None or set(field_names) != set(pde_spec.boundary_fields):
            return False
        for field_name in field_names:
            field_spec = pde_spec.boundary_fields[field_name]
            if self.field_shapes and field_spec.shape not in self.field_shapes:
                return False
            if set(self.operators) - set(field_spec.operators):
                return False
        return True


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

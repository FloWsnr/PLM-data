"""Domain registry, geometry contracts, and sampling-facing metadata."""

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import ufl
from dolfinx import mesh

from plm_data.boundary_conditions import (
    COMMON_BOUNDARY_FAMILIES as _COMMON_BOUNDARY_FAMILIES,
    get_boundary_family_spec,
    has_boundary_family_spec,
)
from plm_data.initial_conditions import (
    COMMON_SCALAR_INITIAL_CONDITION_FAMILIES as _COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
    get_initial_condition_spec,
    has_initial_condition_spec,
)


@dataclass
class PeriodicBoundaryMap:
    """Resolved geometric map for one named periodic boundary pair."""

    name: str
    slave_boundary: str
    master_boundary: str
    matrix: np.ndarray
    offset: np.ndarray
    group_id: str
    slave_selector: Callable[[np.ndarray, float], np.ndarray] | None = None

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Map slave-side coordinates to master-side coordinates."""
        gdim = self.matrix.shape[0]
        out = x.copy()
        out[:gdim, :] = self.matrix @ x[:gdim, :] + self.offset[:, None]
        return out

    def on_slave(self, x: np.ndarray, tol: float) -> np.ndarray:
        """Return a mask selecting points on the slave side."""
        if self.slave_selector is None:
            return np.ones(x.shape[1], dtype=bool)
        return self.slave_selector(x, tol)


@dataclass
class DomainGeometry:
    """Mesh plus named boundary identification."""

    mesh: mesh.Mesh
    facet_tags: mesh.MeshTags  # type: ignore[reportInvalidTypeForm]
    boundary_names: dict[str, int]
    ds: ufl.Measure  # type: ignore[reportInvalidTypeForm]
    periodic_maps: dict[frozenset[str], PeriodicBoundaryMap] = field(
        default_factory=dict
    )

    @property
    def has_periodic_maps(self) -> bool:
        """Return whether the domain exposes any periodic pair maps."""
        return bool(self.periodic_maps)

    def periodic_map(self, side_a: str, side_b: str) -> PeriodicBoundaryMap:
        """Return the resolved periodic map for one side pair."""
        key = frozenset({side_a, side_b})
        if key not in self.periodic_maps:
            raise KeyError(
                f"Domain does not define a periodic map for boundary pair "
                f"{sorted(key)}."
            )
        return self.periodic_maps[key]


@dataclass(frozen=True)
class DomainParameterSpec:
    """Sampling-facing declaration for one domain parameter."""

    name: str
    kind: str
    description: str = ""
    length: int | None = None
    hard_min: float | int | None = None
    hard_max: float | int | None = None
    sampling_min: float | int | None = None
    sampling_max: float | int | None = None


@dataclass(frozen=True)
class DomainSpec:
    """Domain capabilities used by config validation and random sampling."""

    name: str
    dimension: int
    parameters: dict[str, DomainParameterSpec]
    boundary_names: tuple[str, ...]
    boundary_roles: dict[str, tuple[str, ...]]
    dynamic_boundary_patterns: tuple[str, ...] = ()
    periodic_pairs: tuple[tuple[str, str], ...] = ()
    allowed_boundary_families: tuple[str, ...] = ()
    allowed_initial_condition_families: tuple[str, ...] = ()
    coordinate_regions: tuple[str, ...] = ("interior",)
    description: str = ""

    def __post_init__(self) -> None:
        boundary_names = set(self.boundary_names)
        if "all" not in self.boundary_roles:
            raise ValueError(f"Domain spec '{self.name}' must define role 'all'.")
        for role, names in self.boundary_roles.items():
            unknown = set(names) - boundary_names
            if unknown:
                raise ValueError(
                    f"Domain spec '{self.name}' role '{role}' references unknown "
                    f"boundaries {sorted(unknown)}."
                )
        for side_a, side_b in self.periodic_pairs:
            unknown = {side_a, side_b} - boundary_names
            if unknown:
                raise ValueError(
                    f"Domain spec '{self.name}' periodic pair "
                    f"{(side_a, side_b)!r} references unknown boundaries "
                    f"{sorted(unknown)}."
                )
        for name, parameter in self.parameters.items():
            if name != parameter.name:
                raise ValueError(
                    f"Domain spec '{self.name}' parameter key '{name}' does not "
                    f"match DomainParameterSpec.name '{parameter.name}'."
                )
        for family_name in self.allowed_boundary_families:
            if not has_boundary_family_spec(family_name):
                raise ValueError(
                    f"Domain spec '{self.name}' references unknown boundary-condition "
                    f"family '{family_name}'."
                )
            family_spec = get_boundary_family_spec(family_name)
            if not family_spec.is_compatible_with_domain(self):
                raise ValueError(
                    f"Boundary-condition family '{family_name}' is not compatible "
                    f"with domain spec '{self.name}'."
                )
        for family_name in self.allowed_initial_condition_families:
            if not has_initial_condition_spec(family_name):
                raise ValueError(
                    f"Domain spec '{self.name}' references unknown initial-condition "
                    f"family '{family_name}'."
                )
            family_spec = get_initial_condition_spec(family_name)
            if not family_spec.is_compatible_with_domain(self):
                raise ValueError(
                    f"Initial-condition family '{family_name}' is not compatible "
                    f"with domain spec '{self.name}'."
                )


DomainFactory = Callable[[Any], DomainGeometry]
GmshDomainBuilder = Callable[[Any, Any], None]


@dataclass(frozen=True)
class GmshDomainSpec:
    """Registered Gmsh model builder and mesh dimension for one domain."""

    dimension: int
    builder: GmshDomainBuilder


_DOMAIN_REGISTRY: dict[str, DomainFactory] = {}
_DOMAIN_SPEC_REGISTRY: dict[str, DomainSpec] = {}
_GMSH_DOMAIN_REGISTRY: dict[str, GmshDomainSpec] = {}
_DOMAIN_FACTORY_MODULES_LOADED = False

COMMON_SCALAR_INITIAL_CONDITION_FAMILIES = _COMMON_SCALAR_INITIAL_CONDITION_FAMILIES
COMMON_BOUNDARY_FAMILIES = _COMMON_BOUNDARY_FAMILIES


def register_domain(name: str) -> Callable[[DomainFactory], DomainFactory]:
    """Register a domain factory under a config-facing type name."""

    def decorator(factory: DomainFactory) -> DomainFactory:
        _DOMAIN_REGISTRY[name] = factory
        return factory

    return decorator


def register_domain_spec(spec: DomainSpec) -> DomainSpec:
    """Register sampling-facing metadata for one domain."""
    _DOMAIN_SPEC_REGISTRY[spec.name] = spec
    return spec


def list_domain_specs() -> dict[str, DomainSpec]:
    """Return all registered domain specs."""
    return dict(_DOMAIN_SPEC_REGISTRY)


def get_domain_spec(name: str) -> DomainSpec:
    """Return the domain spec registered under one config-facing name."""
    if name not in _DOMAIN_SPEC_REGISTRY:
        available = ", ".join(sorted(_DOMAIN_SPEC_REGISTRY))
        raise ValueError(f"Unknown domain spec '{name}'. Available: {available}")
    return _DOMAIN_SPEC_REGISTRY[name]


def _ensure_domain_factory_modules_loaded() -> None:
    """Load domain factory modules for registration side effects."""
    global _DOMAIN_FACTORY_MODULES_LOADED
    if _DOMAIN_FACTORY_MODULES_LOADED:
        return
    _DOMAIN_FACTORY_MODULES_LOADED = True

    # Import factories lazily so callers can inspect domain specs without pulling
    # in all mesh-generation dependencies.
    domains_package = importlib.import_module("plm_data.domains")
    factory_loader = getattr(domains_package, "_load_domain_factory_modules")
    factory_loader()


def list_domains() -> list[str]:
    """Return the registered domain factory names."""
    _ensure_domain_factory_modules_loaded()
    return sorted(_DOMAIN_REGISTRY)


def create_domain(domain: Any) -> DomainGeometry:
    """Create a mesh with tagged boundaries from a domain configuration."""
    if domain.type not in _DOMAIN_REGISTRY:
        _ensure_domain_factory_modules_loaded()
    if domain.type not in _DOMAIN_REGISTRY:
        raise ValueError(
            f"Unknown domain type: '{domain.type}'. "
            f"Available: {', '.join(list_domains())}"
        )
    return _DOMAIN_REGISTRY[domain.type](domain)


def register_gmsh_domain(
    name: str,
    *,
    dimension: int,
) -> Callable[[GmshDomainBuilder], GmshDomainBuilder]:
    """Register one reusable Gmsh model builder under a domain name."""

    def decorator(builder: GmshDomainBuilder) -> GmshDomainBuilder:
        _GMSH_DOMAIN_REGISTRY[name] = GmshDomainSpec(
            dimension=dimension,
            builder=builder,
        )
        return builder

    return decorator


def is_gmsh_domain(name: str) -> bool:
    """Return whether the domain has a reusable Gmsh builder."""
    if name not in _GMSH_DOMAIN_REGISTRY:
        _ensure_domain_factory_modules_loaded()
    return name in _GMSH_DOMAIN_REGISTRY


def get_gmsh_domain_dimension(name: str) -> int:
    """Return the registered Gmsh mesh dimension for one domain."""
    if name not in _GMSH_DOMAIN_REGISTRY:
        _ensure_domain_factory_modules_loaded()
    if name not in _GMSH_DOMAIN_REGISTRY:
        raise ValueError(f"Domain '{name}' does not expose a Gmsh builder.")
    return _GMSH_DOMAIN_REGISTRY[name].dimension


def build_gmsh_domain_model(domain: Any, model: Any) -> None:
    """Populate the active Gmsh model with one tagged built-in domain."""
    if domain.type not in _GMSH_DOMAIN_REGISTRY:
        _ensure_domain_factory_modules_loaded()
    if domain.type not in _GMSH_DOMAIN_REGISTRY:
        raise ValueError(f"Domain '{domain.type}' does not expose a Gmsh builder.")
    _GMSH_DOMAIN_REGISTRY[domain.type].builder(model, domain)


def register_gmsh_planar_domain(
    name: str,
) -> Callable[[GmshDomainBuilder], GmshDomainBuilder]:
    """Register one reusable 2D Gmsh builder under a domain name."""
    return register_gmsh_domain(name, dimension=2)


def is_gmsh_planar_domain(name: str) -> bool:
    """Return whether the domain has a reusable planar Gmsh builder."""
    if not is_gmsh_domain(name):
        return False
    return get_gmsh_domain_dimension(name) == 2


def build_gmsh_planar_domain_model(domain: Any, model: Any) -> None:
    """Populate the active Gmsh model with one tagged 2D built-in domain."""
    if not is_gmsh_planar_domain(domain.type):
        raise ValueError(
            f"Domain '{domain.type}' does not expose a planar Gmsh builder."
        )
    build_gmsh_domain_model(domain, model)

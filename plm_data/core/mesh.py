"""Mesh creation helpers with a pluggable domain registry."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import ufl
from dolfinx import mesh
from dolfinx.mesh import CellType, GhostMode, locate_entities_boundary, meshtags
from mpi4py import MPI

from plm_data.core.config import DomainConfig


@dataclass
class DomainGeometry:
    """Mesh plus named boundary identification."""

    mesh: mesh.Mesh
    facet_tags: mesh.MeshTags  # type: ignore[reportInvalidTypeForm]
    boundary_names: dict[str, int]
    ds: ufl.Measure  # type: ignore[reportInvalidTypeForm]
    periodic_axes: tuple[int, ...] = ()
    axis_bounds: tuple[tuple[float, float], ...] = ()

    @property
    def has_periodic_axes(self) -> bool:
        """Return whether any periodic directions are configured."""
        return bool(self.periodic_axes)


DomainFactory = Callable[[DomainConfig], DomainGeometry]
_DOMAIN_REGISTRY: dict[str, DomainFactory] = {}


def register_domain(name: str) -> Callable[[DomainFactory], DomainFactory]:
    """Register a domain factory under a config-facing type name."""

    def decorator(factory: DomainFactory) -> DomainFactory:
        _DOMAIN_REGISTRY[name] = factory
        return factory

    return decorator


def list_domains() -> list[str]:
    """Return the registered domain types."""
    return sorted(_DOMAIN_REGISTRY)


def _require_param(params: dict, key: str, domain_type: str):
    """Require a domain parameter, raising a clear error if missing."""
    if key not in params:
        raise ValueError(
            f"Missing required parameter '{key}' for domain type '{domain_type}'. "
            f"Got params: {params}"
        )
    return params[key]


def create_domain(domain: DomainConfig) -> DomainGeometry:
    """Create a mesh with tagged boundaries from a domain configuration."""
    if domain.type not in _DOMAIN_REGISTRY:
        raise ValueError(
            f"Unknown domain type: '{domain.type}'. "
            f"Available: {', '.join(list_domains())}"
        )
    return _DOMAIN_REGISTRY[domain.type](domain)


def _tag_boundaries(
    msh: mesh.Mesh, predicates: dict[str, Callable]
) -> tuple[mesh.MeshTags, dict[str, int], ufl.Measure]:
    """Tag boundary facets using geometric predicates."""
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)

    boundary_names: dict[str, int] = {}
    all_facets = []
    all_tags = []

    for tag_idx, (name, marker) in enumerate(predicates.items(), start=1):
        boundary_names[name] = tag_idx
        facets = locate_entities_boundary(msh, fdim, marker)
        all_facets.append(facets)
        all_tags.append(np.full_like(facets, tag_idx))

    if all_facets:
        facet_indices = np.concatenate(all_facets)
        tag_values = np.concatenate(all_tags)
        sort_order = np.argsort(facet_indices)
        facet_indices = facet_indices[sort_order]
        tag_values = tag_values[sort_order]
    else:
        facet_indices = np.empty(0, dtype=np.int32)
        tag_values = np.empty(0, dtype=np.int32)

    ft = meshtags(msh, fdim, facet_indices, tag_values)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    return ft, boundary_names, ds


@register_domain("interval")
def _create_interval(domain: DomainConfig) -> DomainGeometry:
    p = domain.params
    size = _require_param(p, "size", domain.type)
    res = _require_param(p, "mesh_resolution", domain.type)
    length = float(size[0] if isinstance(size, list) else size)
    nx = int(res[0] if isinstance(res, list) else res)

    msh = mesh.create_interval(
        comm=MPI.COMM_WORLD,
        nx=nx,
        points=[0.0, length],
        ghost_mode=GhostMode.shared_facet,
    )

    predicates = {
        "x-": lambda x, lim=0.0: np.isclose(x[0], lim),
        "x+": lambda x, lim=length: np.isclose(x[0], lim),
    }
    ft, boundary_names, ds = _tag_boundaries(msh, predicates)
    return DomainGeometry(
        mesh=msh,
        facet_tags=ft,
        boundary_names=boundary_names,
        ds=ds,
        periodic_axes=domain.periodic_axes,
        axis_bounds=((0.0, length),),
    )


@register_domain("rectangle")
def _create_rectangle(domain: DomainConfig) -> DomainGeometry:
    p = domain.params
    size = _require_param(p, "size", domain.type)
    res = _require_param(p, "mesh_resolution", domain.type)
    Lx, Ly = size[0], size[1]

    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (Lx, Ly)),
        n=(res[0], res[1]),
        cell_type=CellType.triangle,
        ghost_mode=GhostMode.shared_facet,
    )

    predicates = {
        "x-": lambda x, lim=0.0: np.isclose(x[0], lim),
        "x+": lambda x, lim=Lx: np.isclose(x[0], lim),
        "y-": lambda x, lim=0.0: np.isclose(x[1], lim),
        "y+": lambda x, lim=Ly: np.isclose(x[1], lim),
    }
    ft, boundary_names, ds = _tag_boundaries(msh, predicates)
    return DomainGeometry(
        mesh=msh,
        facet_tags=ft,
        boundary_names=boundary_names,
        ds=ds,
        periodic_axes=domain.periodic_axes,
        axis_bounds=((0.0, Lx), (0.0, Ly)),
    )


@register_domain("box")
def _create_box(domain: DomainConfig) -> DomainGeometry:
    p = domain.params
    size = _require_param(p, "size", domain.type)
    res = _require_param(p, "mesh_resolution", domain.type)
    Lx, Ly, Lz = size[0], size[1], size[2]

    msh = mesh.create_box(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0, 0.0), (Lx, Ly, Lz)),  # type: ignore[reportArgumentType]
        n=(res[0], res[1], res[2]),
        cell_type=CellType.tetrahedron,
        ghost_mode=GhostMode.shared_facet,
    )

    predicates = {
        "x-": lambda x, lim=0.0: np.isclose(x[0], lim),
        "x+": lambda x, lim=Lx: np.isclose(x[0], lim),
        "y-": lambda x, lim=0.0: np.isclose(x[1], lim),
        "y+": lambda x, lim=Ly: np.isclose(x[1], lim),
        "z-": lambda x, lim=0.0: np.isclose(x[2], lim),
        "z+": lambda x, lim=Lz: np.isclose(x[2], lim),
    }
    ft, boundary_names, ds = _tag_boundaries(msh, predicates)
    return DomainGeometry(
        mesh=msh,
        facet_tags=ft,
        boundary_names=boundary_names,
        ds=ds,
        periodic_axes=domain.periodic_axes,
        axis_bounds=((0.0, Lx), (0.0, Ly), (0.0, Lz)),
    )

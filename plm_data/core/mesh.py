"""Mesh creation helpers with boundary tagging."""

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
    """Mesh plus named boundary identification.

    Built-in domains (rectangle, box) auto-generate boundary names from
    geometry (x-, x+, y-, y+, etc.). Future Gmsh domains will populate
    boundary_names from physical groups.
    """

    mesh: mesh.Mesh
    facet_tags: mesh.MeshTags  # type: ignore[reportInvalidTypeForm]
    boundary_names: dict[str, int]  # name → tag integer
    ds: ufl.Measure  # type: ignore[reportInvalidTypeForm]  # ds(tag) integrates over that boundary


def _require_param(params: dict, key: str, domain_type: str):
    """Require a domain parameter, raising a clear error if missing."""
    if key not in params:
        raise ValueError(
            f"Missing required parameter '{key}' for domain type '{domain_type}'. "
            f"Got params: {params}"
        )
    return params[key]


def create_domain(domain: DomainConfig) -> DomainGeometry:
    """Create a mesh with tagged boundaries from a domain configuration.

    Dispatches on domain.type. All required params must be in domain.params.
    """
    if domain.type == "rectangle":
        return _create_rectangle(domain)
    elif domain.type == "box":
        return _create_box(domain)
    else:
        raise ValueError(
            f"Unknown domain type: '{domain.type}'. Available: rectangle, box"
        )


def _tag_boundaries(
    msh: mesh.Mesh, predicates: dict[str, Callable]
) -> tuple[mesh.MeshTags, dict[str, int], ufl.Measure]:
    """Tag boundary facets using geometric predicates.

    Args:
        msh: The mesh.
        predicates: Mapping from boundary name to a marker function
            with signature (x: ndarray) -> bool ndarray.

    Returns:
        (facet_tags, boundary_names, ds) tuple.
    """
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)

    boundary_names = {}
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
        # Sort by facet index (required by meshtags)
        sort_order = np.argsort(facet_indices)
        facet_indices = facet_indices[sort_order]
        tag_values = tag_values[sort_order]
    else:
        facet_indices = np.empty(0, dtype=np.int32)
        tag_values = np.empty(0, dtype=np.int32)

    ft = meshtags(msh, fdim, facet_indices, tag_values)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    return ft, boundary_names, ds


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
    return DomainGeometry(mesh=msh, facet_tags=ft, boundary_names=boundary_names, ds=ds)


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
    return DomainGeometry(mesh=msh, facet_tags=ft, boundary_names=boundary_names, ds=ds)

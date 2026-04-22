"""Compatibility imports for domain mesh generation."""

from dolfinx import mesh as _mesh
from dolfinx.mesh import GhostMode as _GhostMode

from plm_data.domains import (
    DomainGeometry,
    PeriodicBoundaryMap,
    build_gmsh_domain_model,
    build_gmsh_planar_domain_model,
    create_domain,
    get_gmsh_domain_dimension,
    is_gmsh_domain,
    is_gmsh_planar_domain,
    list_domains,
)
from plm_data.domains.gmsh import model_to_mesh_shared_facet

_model_to_mesh_shared_facet = model_to_mesh_shared_facet
mesh = _mesh
GhostMode = _GhostMode

__all__ = [
    "DomainGeometry",
    "GhostMode",
    "PeriodicBoundaryMap",
    "build_gmsh_domain_model",
    "build_gmsh_planar_domain_model",
    "create_domain",
    "get_gmsh_domain_dimension",
    "is_gmsh_domain",
    "is_gmsh_planar_domain",
    "list_domains",
    "mesh",
]

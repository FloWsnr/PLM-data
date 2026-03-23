"""Mesh creation helpers."""

from dolfinx import mesh
from dolfinx.mesh import CellType, GhostMode
from mpi4py import MPI

from plm_data.core.config import DomainConfig


def _require_param(params: dict, key: str, domain_type: str):
    """Require a domain parameter, raising a clear error if missing."""
    if key not in params:
        raise ValueError(
            f"Missing required parameter '{key}' for domain type '{domain_type}'. "
            f"Got params: {params}"
        )
    return params[key]


def create_mesh(domain: DomainConfig) -> mesh.Mesh:
    """Create a mesh from a domain configuration.

    Dispatches on domain.type. All required params must be in domain.params.
    """
    if domain.type == "rectangle":
        return _create_rectangle(domain)
    elif domain.type == "box":
        return _create_box(domain)
    else:
        raise ValueError(
            f"Unknown domain type: '{domain.type}'. "
            f"Available: rectangle, box"
        )


def _create_rectangle(domain: DomainConfig) -> mesh.Mesh:
    p = domain.params
    size = _require_param(p, "size", domain.type)
    res = _require_param(p, "mesh_resolution", domain.type)
    return mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (size[0], size[1])),
        n=(res[0], res[1]),
        cell_type=CellType.triangle,
        ghost_mode=GhostMode.shared_facet,
    )


def _create_box(domain: DomainConfig) -> mesh.Mesh:
    p = domain.params
    size = _require_param(p, "size", domain.type)
    res = _require_param(p, "mesh_resolution", domain.type)
    return mesh.create_box(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0, 0.0), (size[0], size[1], size[2])),
        n=(res[0], res[1], res[2]),
        cell_type=CellType.tetrahedron,
        ghost_mode=GhostMode.shared_facet,
    )

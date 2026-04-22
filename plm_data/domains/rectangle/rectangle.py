"""Rectangle domain factory."""

import numpy as np
from dolfinx import mesh
from dolfinx.mesh import CellType, GhostMode
from mpi4py import MPI

from plm_data.core.config import DomainConfig
from plm_data.domains.base import DomainGeometry, register_domain
from plm_data.domains.helpers import (
    builtin_periodic_map,
    compile_config_periodic_maps,
    merge_periodic_maps,
    require_param,
    tag_boundaries,
)


@register_domain("rectangle")
def create_rectangle(domain: DomainConfig) -> DomainGeometry:
    """Create an axis-aligned 2D rectangle mesh."""
    p = domain.params
    size = require_param(p, "size", domain.type)
    res = require_param(p, "mesh_resolution", domain.type)
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
    ft, boundary_names, ds = tag_boundaries(msh, predicates)
    builtin_maps = merge_periodic_maps(
        builtin_periodic_map(
            name="x",
            group_id="x",
            slave_boundary="x+",
            master_boundary="x-",
            slave_selector=lambda x, tol, lim=Lx: np.isclose(
                x[0], lim, atol=tol, rtol=tol
            ),
            offset=(-Lx, 0.0),
        ),
        builtin_periodic_map(
            name="y",
            group_id="y",
            slave_boundary="y+",
            master_boundary="y-",
            slave_selector=lambda x, tol, lim=Ly: np.isclose(
                x[1], lim, atol=tol, rtol=tol
            ),
            offset=(0.0, -Ly),
        ),
    )
    periodic_maps = merge_periodic_maps(
        builtin_maps,
        compile_config_periodic_maps(domain.periodic_maps),
    )
    return DomainGeometry(
        mesh=msh,
        facet_tags=ft,
        boundary_names=boundary_names,
        ds=ds,
        periodic_maps=periodic_maps,
    )

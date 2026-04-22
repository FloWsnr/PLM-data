"""Gmsh-backed disk domain."""

import numpy as np

from plm_data.core.config import DomainConfig, validate_domain_params
from plm_data.domains.gmsh import (
    register_gmsh_domain_factory,
    tag_single_gmsh_boundary,
)
from plm_data.domains.helpers import require_param


@register_gmsh_domain_factory("disk", dimension=2)
def build_disk_gmsh_model(model, domain: DomainConfig) -> None:
    """Populate the active Gmsh model with a tagged disk."""
    p = domain.params
    center = np.asarray(require_param(p, "center", domain.type), dtype=float)
    radius = float(require_param(p, "radius", domain.type))
    mesh_size = float(require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    disk = model.occ.addDisk(center[0], center[1], 0.0, radius, radius)
    model.occ.synchronize()

    model.addPhysicalGroup(2, [disk], tag=1)
    model.setPhysicalName(2, 1, "surface")
    tag_single_gmsh_boundary(model, [disk], "outer")

    model.mesh.setSize(model.getEntities(0), mesh_size)

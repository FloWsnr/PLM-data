"""Gmsh-backed dumbbell domain."""

import numpy as np

from plm_data.core.config import DomainConfig, validate_domain_params
from plm_data.domains.gmsh import (
    register_gmsh_domain_factory,
    tag_single_gmsh_boundary,
)
from plm_data.domains.helpers import require_param


@register_gmsh_domain_factory("dumbbell", dimension=2)
def build_dumbbell_gmsh_model(model, domain: DomainConfig) -> None:
    """Populate the active Gmsh model with a tagged dumbbell."""
    p = domain.params
    left_center = np.asarray(require_param(p, "left_center", domain.type), dtype=float)
    right_center = np.asarray(
        require_param(p, "right_center", domain.type),
        dtype=float,
    )
    lobe_radius = float(require_param(p, "lobe_radius", domain.type))
    neck_width = float(require_param(p, "neck_width", domain.type))
    mesh_size = float(require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    left_disk = model.occ.addDisk(
        left_center[0],
        left_center[1],
        0.0,
        lobe_radius,
        lobe_radius,
    )
    right_disk = model.occ.addDisk(
        right_center[0],
        right_center[1],
        0.0,
        lobe_radius,
        lobe_radius,
    )
    bridge = model.occ.addRectangle(
        left_center[0],
        left_center[1] - 0.5 * neck_width,
        0.0,
        right_center[0] - left_center[0],
        neck_width,
    )
    model.occ.fuse(
        [(2, left_disk), (2, right_disk)],
        [(2, bridge)],
        removeObject=True,
        removeTool=True,
    )
    model.occ.synchronize()

    surfaces = [entity[1] for entity in model.getEntities(2)]
    model.addPhysicalGroup(2, surfaces, tag=1)
    model.setPhysicalName(2, 1, "surface")
    tag_single_gmsh_boundary(model, surfaces, "outer")

    model.mesh.setSize(model.getEntities(0), mesh_size)

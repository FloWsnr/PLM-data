"""Gmsh-backed annulus domain."""

import numpy as np

from plm_data.core.config import DomainConfig, validate_domain_params
from plm_data.domains.gmsh import register_gmsh_domain_factory
from plm_data.domains.helpers import require_param


@register_gmsh_domain_factory("annulus", dimension=2)
def build_annulus_gmsh_model(model, domain: DomainConfig) -> None:
    """Populate the active Gmsh model with a tagged annulus."""
    p = domain.params
    center = np.asarray(require_param(p, "center", domain.type), dtype=float)
    inner_radius = float(require_param(p, "inner_radius", domain.type))
    outer_radius = float(require_param(p, "outer_radius", domain.type))
    mesh_size = float(require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    outer_disk = model.occ.addDisk(
        center[0],
        center[1],
        0.0,
        outer_radius,
        outer_radius,
    )
    inner_disk = model.occ.addDisk(
        center[0],
        center[1],
        0.0,
        inner_radius,
        inner_radius,
    )
    model.occ.cut([(2, outer_disk)], [(2, inner_disk)])
    model.occ.synchronize()

    surfaces = [entity[1] for entity in model.getEntities(2)]
    model.addPhysicalGroup(2, surfaces, tag=1)
    model.setPhysicalName(2, 1, "surface")

    boundary = model.getBoundary(
        [(2, surface_tag) for surface_tag in surfaces],
        oriented=False,
    )
    r_mid = inner_radius + outer_radius
    inner_curves: list[int] = []
    outer_curves: list[int] = []
    for dim, tag in boundary:
        bb = model.occ.getBoundingBox(dim, tag)
        extent = max(bb[3] - bb[0], bb[4] - bb[1])
        if extent < r_mid:
            inner_curves.append(tag)
        else:
            outer_curves.append(tag)

    model.addPhysicalGroup(1, inner_curves, tag=1)
    model.setPhysicalName(1, 1, "inner")
    model.addPhysicalGroup(1, outer_curves, tag=2)
    model.setPhysicalName(1, 2, "outer")

    model.mesh.setSize(model.getEntities(0), mesh_size)

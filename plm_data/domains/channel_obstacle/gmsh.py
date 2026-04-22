"""Gmsh-backed channel-with-obstacle domain."""

import numpy as np

from plm_data.core.config import DomainConfig, validate_domain_params
from plm_data.domains.gmsh import (
    add_named_gmsh_boundaries,
    classify_rectangular_channel_hole_boundaries,
    register_gmsh_domain_factory,
)
from plm_data.domains.helpers import require_param


@register_gmsh_domain_factory("channel_obstacle", dimension=2)
def build_channel_obstacle_gmsh_model(model, domain: DomainConfig) -> None:
    """Populate the active Gmsh model with a tagged obstacle channel."""
    p = domain.params
    length = float(require_param(p, "length", domain.type))
    height = float(require_param(p, "height", domain.type))
    obstacle_center = np.asarray(
        require_param(p, "obstacle_center", domain.type),
        dtype=float,
    )
    obstacle_radius = float(require_param(p, "obstacle_radius", domain.type))
    mesh_size = float(require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    channel = model.occ.addRectangle(0.0, 0.0, 0.0, length, height)
    obstacle = model.occ.addDisk(
        obstacle_center[0],
        obstacle_center[1],
        0.0,
        obstacle_radius,
        obstacle_radius,
    )
    model.occ.cut(
        [(2, channel)],
        [(2, obstacle)],
        removeObject=True,
        removeTool=True,
    )
    model.occ.synchronize()

    surfaces = [entity[1] for entity in model.getEntities(2)]
    model.addPhysicalGroup(2, surfaces, tag=1)
    model.setPhysicalName(2, 1, "surface")

    tol = max(mesh_size, 1.0e-8)
    inlet, outlet, walls, obstacle_curves = (
        classify_rectangular_channel_hole_boundaries(
            model,
            surfaces,
            length=length,
            height=height,
            tol=tol,
        )
    )
    add_named_gmsh_boundaries(
        model,
        (
            ("inlet", inlet),
            ("outlet", outlet),
            ("walls", walls),
            ("obstacle", obstacle_curves),
        ),
        domain_name="Channel-obstacle",
    )

    model.mesh.setSize(model.getEntities(0), mesh_size)

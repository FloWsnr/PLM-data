"""Gmsh-backed side-cavity channel domain."""

import numpy as np

from plm_data.core.config import DomainConfig, validate_domain_params
from plm_data.domains.gmsh import fuse_planar_surfaces, register_gmsh_domain_factory
from plm_data.domains.helpers import require_param


@register_gmsh_domain_factory("side_cavity_channel", dimension=2)
def build_side_cavity_channel_gmsh_model(model, domain: DomainConfig) -> None:
    """Populate the active Gmsh model with a tagged side-cavity channel."""
    p = domain.params
    length = float(require_param(p, "length", domain.type))
    height = float(require_param(p, "height", domain.type))
    cavity_width = float(require_param(p, "cavity_width", domain.type))
    cavity_depth = float(require_param(p, "cavity_depth", domain.type))
    cavity_center_x = float(require_param(p, "cavity_center_x", domain.type))
    mesh_size = float(require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    cavity_left = cavity_center_x - 0.5 * cavity_width

    channel = model.occ.addRectangle(0.0, 0.0, 0.0, length, height)
    cavity = model.occ.addRectangle(
        cavity_left,
        height,
        0.0,
        cavity_width,
        cavity_depth,
    )
    surfaces = fuse_planar_surfaces(model, [channel, cavity])
    model.addPhysicalGroup(2, surfaces, tag=1)
    model.setPhysicalName(2, 1, "surface")

    inlet: list[int] = []
    outlet: list[int] = []
    walls: list[int] = []
    boundary_tol = 1.0e-6
    boundary = model.getBoundary(
        [(2, surface_tag) for surface_tag in surfaces],
        oriented=False,
    )
    for dim, tag in boundary:
        if dim != 1:
            continue
        bb = model.occ.getBoundingBox(dim, tag)
        x_min, _, _, x_max, _, _ = bb
        if np.isclose(x_min, 0.0, atol=boundary_tol) and np.isclose(
            x_max,
            0.0,
            atol=boundary_tol,
        ):
            inlet.append(tag)
        elif np.isclose(x_min, length, atol=boundary_tol) and np.isclose(
            x_max,
            length,
            atol=boundary_tol,
        ):
            outlet.append(tag)
        else:
            walls.append(tag)

    for physical_tag, (name, curve_tags) in enumerate(
        (
            ("inlet", inlet),
            ("outlet", outlet),
            ("walls", walls),
        ),
        start=1,
    ):
        if not curve_tags:
            raise AssertionError(
                f"Side-cavity channel domain produced no curves for '{name}'."
            )
        model.addPhysicalGroup(1, curve_tags, tag=physical_tag)
        model.setPhysicalName(1, physical_tag, name)

    model.mesh.setSize(model.getEntities(0), mesh_size)

"""Gmsh-backed L-shaped domain."""

import numpy as np

from plm_data.core.config import DomainConfig, validate_domain_params
from plm_data.domains.gmsh import fuse_planar_surfaces, register_gmsh_domain_factory
from plm_data.domains.helpers import require_param


@register_gmsh_domain_factory("l_shape", dimension=2)
def build_l_shape_gmsh_model(model, domain: DomainConfig) -> None:
    """Populate the active Gmsh model with a tagged L-shaped domain."""
    p = domain.params
    outer_width = float(require_param(p, "outer_width", domain.type))
    outer_height = float(require_param(p, "outer_height", domain.type))
    cutout_width = float(require_param(p, "cutout_width", domain.type))
    cutout_height = float(require_param(p, "cutout_height", domain.type))
    mesh_size = float(require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    reentrant_x = outer_width - cutout_width
    reentrant_y = outer_height - cutout_height

    lower_arm = model.occ.addRectangle(
        0.0,
        0.0,
        0.0,
        outer_width,
        reentrant_y,
    )
    left_arm = model.occ.addRectangle(
        0.0,
        0.0,
        0.0,
        reentrant_x,
        outer_height,
    )
    surfaces = fuse_planar_surfaces(model, [lower_arm, left_arm])
    model.addPhysicalGroup(2, surfaces, tag=1)
    model.setPhysicalName(2, 1, "surface")

    tol = max(1.0e-8, 1.0e-6 * max(outer_width, outer_height))
    outer: list[int] = []
    notch: list[int] = []
    boundary = model.getBoundary(
        [(2, surface_tag) for surface_tag in surfaces],
        oriented=False,
    )
    for dim, tag in boundary:
        if dim != 1:
            continue
        x_min, y_min, _, x_max, y_max, _ = model.occ.getBoundingBox(dim, tag)
        is_notch_vertical = np.isclose(x_min, reentrant_x, atol=tol) and np.isclose(
            x_max,
            reentrant_x,
            atol=tol,
        )
        is_notch_horizontal = np.isclose(y_min, reentrant_y, atol=tol) and np.isclose(
            y_max,
            reentrant_y,
            atol=tol,
        )
        if is_notch_vertical or is_notch_horizontal:
            notch.append(tag)
        else:
            outer.append(tag)

    for physical_tag, (name, curve_tags) in enumerate(
        (
            ("outer", outer),
            ("notch", notch),
        ),
        start=1,
    ):
        if not curve_tags:
            raise AssertionError(f"L-shape domain produced no curves for '{name}'.")
        model.addPhysicalGroup(1, curve_tags, tag=physical_tag)
        model.setPhysicalName(1, physical_tag, name)

    model.mesh.setSize(model.getEntities(0), mesh_size)

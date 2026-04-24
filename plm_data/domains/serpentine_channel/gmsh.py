"""Gmsh-backed serpentine channel domain."""

import numpy as np

from plm_data.domains.gmsh import (
    add_orthogonal_channel_surfaces,
    register_gmsh_domain_factory,
)
from plm_data.domains.helpers import require_param
from plm_data.domains.validation import DomainConfigLike, validate_domain_params


@register_gmsh_domain_factory("serpentine_channel", dimension=2)
def build_serpentine_channel_gmsh_model(model, domain: DomainConfigLike) -> None:
    """Populate the active Gmsh model with a tagged serpentine channel."""
    p = domain.params
    channel_length = float(require_param(p, "channel_length", domain.type))
    lane_spacing = float(require_param(p, "lane_spacing", domain.type))
    n_bends = int(require_param(p, "n_bends", domain.type))
    channel_width = float(require_param(p, "channel_width", domain.type))
    mesh_size = float(require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    centerline_points = [np.array([0.0, 0.0], dtype=float)]
    current_x = 0.0
    current_y = 0.0
    direction = 1.0
    for bend_index in range(n_bends + 1):
        current_x = channel_length if direction > 0.0 else 0.0
        centerline_points.append(np.array([current_x, current_y], dtype=float))
        if bend_index == n_bends:
            break
        current_y += lane_spacing
        centerline_points.append(np.array([current_x, current_y], dtype=float))
        direction *= -1.0

    inlet_center = centerline_points[0]
    outlet_center = centerline_points[-1]
    cap_tolerance = 0.45 * channel_width + 1.0e-8

    surfaces = add_orthogonal_channel_surfaces(
        model,
        centerline_points=centerline_points,
        channel_width=channel_width,
    )
    model.addPhysicalGroup(2, surfaces, tag=1)
    model.setPhysicalName(2, 1, "surface")

    inlet: list[int] = []
    outlet: list[int] = []
    walls: list[int] = []
    boundary = model.getBoundary(
        [(2, surface_tag) for surface_tag in surfaces],
        oriented=False,
    )
    for dim, tag in boundary:
        if dim != 1:
            continue
        center = np.asarray(
            model.occ.getCenterOfMass(dim, tag)[:2],
            dtype=float,
        )
        if np.linalg.norm(center - inlet_center) <= cap_tolerance:
            inlet.append(tag)
            continue
        if np.linalg.norm(center - outlet_center) <= cap_tolerance:
            outlet.append(tag)
            continue
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
                f"Serpentine-channel domain produced no curves for '{name}'."
            )
        model.addPhysicalGroup(1, curve_tags, tag=physical_tag)
        model.setPhysicalName(1, physical_tag, name)

    model.mesh.setSize(model.getEntities(0), mesh_size)

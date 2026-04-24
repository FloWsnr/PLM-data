"""Gmsh-backed Venturi channel domain."""

import numpy as np

from plm_data.domains.gmsh import register_gmsh_domain_factory
from plm_data.domains.helpers import require_param
from plm_data.domains.validation import DomainConfigLike, validate_domain_params


@register_gmsh_domain_factory("venturi_channel", dimension=2)
def build_venturi_channel_gmsh_model(model, domain: DomainConfigLike) -> None:
    """Populate the active Gmsh model with a tagged Venturi channel."""
    p = domain.params
    length = float(require_param(p, "length", domain.type))
    height = float(require_param(p, "height", domain.type))
    throat_height = float(require_param(p, "throat_height", domain.type))
    constriction_center_x = float(
        require_param(p, "constriction_center_x", domain.type)
    )
    constriction_radius = float(require_param(p, "constriction_radius", domain.type))
    mesh_size = float(require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    indentation = 0.5 * (height - throat_height)
    top_center_y = height + constriction_radius - indentation
    bottom_center_y = -constriction_radius + indentation

    channel = model.occ.addRectangle(0.0, 0.0, 0.0, length, height)
    top_cut = model.occ.addDisk(
        constriction_center_x,
        top_center_y,
        0.0,
        constriction_radius,
        constriction_radius,
    )
    bottom_cut = model.occ.addDisk(
        constriction_center_x,
        bottom_center_y,
        0.0,
        constriction_radius,
        constriction_radius,
    )
    model.occ.cut(
        [(2, channel)],
        [(2, top_cut), (2, bottom_cut)],
        removeObject=True,
        removeTool=True,
    )
    model.occ.synchronize()

    surfaces = [entity[1] for entity in model.getEntities(2)]
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
                f"Venturi-channel domain produced no curves for '{name}'."
            )
        model.addPhysicalGroup(1, curve_tags, tag=physical_tag)
        model.setPhysicalName(1, physical_tag, name)

    model.mesh.setSize(model.getEntities(0), mesh_size)

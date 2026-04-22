"""Gmsh-backed porous channel domain."""

from plm_data.core.config import DomainConfig, validate_domain_params
from plm_data.domains.gmsh import (
    add_named_gmsh_boundaries,
    classify_rectangular_channel_hole_boundaries,
    porous_channel_obstacle_centers,
    register_gmsh_domain_factory,
)
from plm_data.domains.helpers import require_param


@register_gmsh_domain_factory("porous_channel", dimension=2)
def build_porous_channel_gmsh_model(model, domain: DomainConfig) -> None:
    """Populate the active Gmsh model with a tagged porous channel."""
    p = domain.params
    length = float(require_param(p, "length", domain.type))
    height = float(require_param(p, "height", domain.type))
    obstacle_radius = float(require_param(p, "obstacle_radius", domain.type))
    n_rows = int(require_param(p, "n_rows", domain.type))
    n_cols = int(require_param(p, "n_cols", domain.type))
    pitch_x = float(require_param(p, "pitch_x", domain.type))
    pitch_y = float(require_param(p, "pitch_y", domain.type))
    x_margin = float(require_param(p, "x_margin", domain.type))
    y_margin = float(require_param(p, "y_margin", domain.type))
    row_shift_fraction = float(require_param(p, "row_shift_fraction", domain.type))
    mesh_size = float(require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    obstacle_centers = porous_channel_obstacle_centers(
        obstacle_radius=obstacle_radius,
        n_rows=n_rows,
        n_cols=n_cols,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        x_margin=x_margin,
        y_margin=y_margin,
        row_shift_fraction=row_shift_fraction,
    )

    channel = model.occ.addRectangle(0.0, 0.0, 0.0, length, height)
    obstacles = [
        model.occ.addDisk(
            center_x,
            center_y,
            0.0,
            obstacle_radius,
            obstacle_radius,
        )
        for center_x, center_y in obstacle_centers
    ]
    model.occ.cut(
        [(2, channel)],
        [(2, obstacle) for obstacle in obstacles],
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
            ("obstacles", obstacle_curves),
        ),
        domain_name="Porous-channel",
    )

    model.mesh.setSize(model.getEntities(0), mesh_size)

"""Gmsh-backed rectangle domain."""

import numpy as np

from plm_data.core.config import DomainConfig, validate_domain_params
from plm_data.domains.base import DomainGeometry, register_domain, register_gmsh_domain
from plm_data.domains.gmsh import add_named_gmsh_boundaries, create_gmsh_domain
from plm_data.domains.helpers import (
    builtin_periodic_map,
    compile_config_periodic_maps,
    merge_periodic_maps,
    require_param,
)


def _size_and_resolution(domain: DomainConfig) -> tuple[float, float, int, int]:
    p = domain.params
    size = require_param(p, "size", domain.type)
    res = require_param(p, "mesh_resolution", domain.type)
    return float(size[0]), float(size[1]), int(res[0]), int(res[1])


@register_gmsh_domain("rectangle", dimension=2)
def build_rectangle_gmsh_model(model, domain: DomainConfig) -> None:
    """Populate the active Gmsh model with a tagged rectangle."""
    validate_domain_params(domain.type, domain.params)
    Lx, Ly, nx, ny = _size_and_resolution(domain)

    surface = model.occ.addRectangle(0.0, 0.0, 0.0, Lx, Ly)
    model.occ.synchronize()

    model.addPhysicalGroup(2, [surface], tag=1)
    model.setPhysicalName(2, 1, "surface")

    tol = max(1.0e-6, 1.0e-8 * max(Lx, Ly))
    x_min: list[int] = []
    x_max: list[int] = []
    y_min: list[int] = []
    y_max: list[int] = []
    boundary = model.getBoundary([(2, surface)], oriented=False)
    for dim, tag in boundary:
        if dim != 1:
            continue
        bb = model.occ.getBoundingBox(dim, tag)
        curve_x_min, curve_y_min, _, curve_x_max, curve_y_max, _ = bb
        if np.isclose(curve_x_min, 0.0, atol=tol) and np.isclose(
            curve_x_max,
            0.0,
            atol=tol,
        ):
            x_min.append(tag)
            model.mesh.setTransfiniteCurve(tag, ny + 1)
        elif np.isclose(curve_x_min, Lx, atol=tol) and np.isclose(
            curve_x_max,
            Lx,
            atol=tol,
        ):
            x_max.append(tag)
            model.mesh.setTransfiniteCurve(tag, ny + 1)
        elif np.isclose(curve_y_min, 0.0, atol=tol) and np.isclose(
            curve_y_max,
            0.0,
            atol=tol,
        ):
            y_min.append(tag)
            model.mesh.setTransfiniteCurve(tag, nx + 1)
        elif np.isclose(curve_y_min, Ly, atol=tol) and np.isclose(
            curve_y_max,
            Ly,
            atol=tol,
        ):
            y_max.append(tag)
            model.mesh.setTransfiniteCurve(tag, nx + 1)

    add_named_gmsh_boundaries(
        model,
        (
            ("x-", x_min),
            ("x+", x_max),
            ("y-", y_min),
            ("y+", y_max),
        ),
        domain_name="Rectangle",
    )
    model.mesh.setTransfiniteSurface(surface)


@register_domain("rectangle")
def create_rectangle(domain: DomainConfig) -> DomainGeometry:
    """Create a tagged 2D rectangle through Gmsh."""
    Lx, Ly, _, _ = _size_and_resolution(domain)
    domain_geom = create_gmsh_domain(domain)
    builtin_maps = merge_periodic_maps(
        builtin_periodic_map(
            name="x",
            group_id="x",
            slave_boundary="x+",
            master_boundary="x-",
            slave_selector=lambda x, tol, lim=Lx: np.isclose(
                x[0],
                lim,
                atol=tol,
                rtol=tol,
            ),
            offset=(-Lx, 0.0),
        ),
        builtin_periodic_map(
            name="y",
            group_id="y",
            slave_boundary="y+",
            master_boundary="y-",
            slave_selector=lambda x, tol, lim=Ly: np.isclose(
                x[1],
                lim,
                atol=tol,
                rtol=tol,
            ),
            offset=(0.0, -Ly),
        ),
    )
    domain_geom.periodic_maps = merge_periodic_maps(
        builtin_maps,
        compile_config_periodic_maps(domain.periodic_maps),
    )
    return domain_geom

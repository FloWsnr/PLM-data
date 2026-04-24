"""Gmsh-backed parallelogram domain."""

import numpy as np

from plm_data.domains.base import DomainGeometry, register_domain, register_gmsh_domain
from plm_data.domains.gmsh import create_gmsh_domain
from plm_data.domains.helpers import (
    builtin_periodic_map,
    compile_config_periodic_maps,
    merge_periodic_maps,
    require_param,
)
from plm_data.domains.validation import DomainConfigLike, validate_domain_params


def _local_coordinates(
    x: np.ndarray,
    origin: np.ndarray,
    inverse_basis: np.ndarray,
) -> np.ndarray:
    """Return local parallelogram coordinates for physical points."""
    return inverse_basis @ (x[:2, :] - origin[:, None])


def _params(
    domain: DomainConfigLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    p = domain.params
    origin = np.asarray(require_param(p, "origin", domain.type), dtype=float)
    axis_x = np.asarray(require_param(p, "axis_x", domain.type), dtype=float)
    axis_y = np.asarray(require_param(p, "axis_y", domain.type), dtype=float)
    res = require_param(p, "mesh_resolution", domain.type)
    return origin, axis_x, axis_y, int(res[0]), int(res[1])


@register_gmsh_domain("parallelogram", dimension=2)
def build_parallelogram_gmsh_model(model, domain: DomainConfigLike) -> None:
    """Populate the active Gmsh model with a tagged parallelogram."""
    validate_domain_params(domain.type, domain.params)
    origin, axis_x, axis_y, nx, ny = _params(domain)
    p0_xy = origin
    p1_xy = origin + axis_x
    p2_xy = origin + axis_x + axis_y
    p3_xy = origin + axis_y

    p0 = model.occ.addPoint(float(p0_xy[0]), float(p0_xy[1]), 0.0)
    p1 = model.occ.addPoint(float(p1_xy[0]), float(p1_xy[1]), 0.0)
    p2 = model.occ.addPoint(float(p2_xy[0]), float(p2_xy[1]), 0.0)
    p3 = model.occ.addPoint(float(p3_xy[0]), float(p3_xy[1]), 0.0)
    y_min = model.occ.addLine(p0, p1)
    x_max = model.occ.addLine(p1, p2)
    y_max = model.occ.addLine(p2, p3)
    x_min = model.occ.addLine(p3, p0)
    loop = model.occ.addCurveLoop([y_min, x_max, y_max, x_min])
    surface = model.occ.addPlaneSurface([loop])
    model.occ.synchronize()

    model.addPhysicalGroup(2, [surface], tag=1)
    model.setPhysicalName(2, 1, "surface")
    for physical_tag, (name, curve_tags) in enumerate(
        (
            ("x-", [x_min]),
            ("x+", [x_max]),
            ("y-", [y_min]),
            ("y+", [y_max]),
        ),
        start=1,
    ):
        model.addPhysicalGroup(1, curve_tags, tag=physical_tag)
        model.setPhysicalName(1, physical_tag, name)

    model.mesh.setTransfiniteCurve(y_min, nx + 1)
    model.mesh.setTransfiniteCurve(y_max, nx + 1)
    model.mesh.setTransfiniteCurve(x_min, ny + 1)
    model.mesh.setTransfiniteCurve(x_max, ny + 1)
    model.mesh.setTransfiniteSurface(surface)


@register_domain("parallelogram")
def create_parallelogram(domain: DomainConfigLike) -> DomainGeometry:
    """Create a tagged 2D parallelogram through Gmsh."""
    origin, axis_x, axis_y, _, _ = _params(domain)
    domain_geom = create_gmsh_domain(domain)

    inverse_basis = np.linalg.inv(np.column_stack((axis_x, axis_y)))
    builtin_maps = merge_periodic_maps(
        builtin_periodic_map(
            name="x",
            group_id="x",
            slave_boundary="x+",
            master_boundary="x-",
            slave_selector=lambda x, tol: np.isclose(
                _local_coordinates(x, origin, inverse_basis)[0],
                1.0,
                atol=tol,
                rtol=tol,
            ),
            offset=tuple(-axis_x),
        ),
        builtin_periodic_map(
            name="y",
            group_id="y",
            slave_boundary="y+",
            master_boundary="y-",
            slave_selector=lambda x, tol: np.isclose(
                _local_coordinates(x, origin, inverse_basis)[1],
                1.0,
                atol=tol,
                rtol=tol,
            ),
            offset=tuple(-axis_y),
        ),
    )
    domain_geom.periodic_maps = merge_periodic_maps(
        builtin_maps,
        compile_config_periodic_maps(domain.periodic_maps),
    )
    return domain_geom

"""Gmsh-backed box domain."""

import numpy as np

from plm_data.core.config import DomainConfig, validate_domain_params
from plm_data.domains.base import DomainGeometry, register_domain, register_gmsh_domain
from plm_data.domains.gmsh import create_gmsh_domain
from plm_data.domains.helpers import (
    builtin_periodic_map,
    compile_config_periodic_maps,
    merge_periodic_maps,
    require_param,
)


def _size_and_resolution(
    domain: DomainConfig,
) -> tuple[float, float, float, int, int, int]:
    p = domain.params
    size = require_param(p, "size", domain.type)
    res = require_param(p, "mesh_resolution", domain.type)
    return (
        float(size[0]),
        float(size[1]),
        float(size[2]),
        int(res[0]),
        int(res[1]),
        int(res[2]),
    )


def _add_box_boundaries(
    model,
    volume: int,
    *,
    Lx: float,
    Ly: float,
    Lz: float,
) -> list[int]:
    tol = max(1.0e-6, 1.0e-8 * max(Lx, Ly, Lz))
    boundary_groups: dict[str, list[int]] = {
        "x-": [],
        "x+": [],
        "y-": [],
        "y+": [],
        "z-": [],
        "z+": [],
    }
    boundary = model.getBoundary([(3, volume)], oriented=False)
    surfaces: list[int] = []
    for dim, tag in boundary:
        if dim != 2:
            continue
        surfaces.append(tag)
        x_min, y_min, z_min, x_max, y_max, z_max = model.occ.getBoundingBox(dim, tag)
        if np.isclose(x_min, 0.0, atol=tol) and np.isclose(x_max, 0.0, atol=tol):
            boundary_groups["x-"].append(tag)
        elif np.isclose(x_min, Lx, atol=tol) and np.isclose(x_max, Lx, atol=tol):
            boundary_groups["x+"].append(tag)
        elif np.isclose(y_min, 0.0, atol=tol) and np.isclose(y_max, 0.0, atol=tol):
            boundary_groups["y-"].append(tag)
        elif np.isclose(y_min, Ly, atol=tol) and np.isclose(y_max, Ly, atol=tol):
            boundary_groups["y+"].append(tag)
        elif np.isclose(z_min, 0.0, atol=tol) and np.isclose(z_max, 0.0, atol=tol):
            boundary_groups["z-"].append(tag)
        elif np.isclose(z_min, Lz, atol=tol) and np.isclose(z_max, Lz, atol=tol):
            boundary_groups["z+"].append(tag)

    for physical_tag, name in enumerate(boundary_groups, start=1):
        group_surfaces = boundary_groups[name]
        if not group_surfaces:
            raise AssertionError(f"Box domain produced no surfaces for '{name}'.")
        model.addPhysicalGroup(2, group_surfaces, tag=physical_tag)
        model.setPhysicalName(2, physical_tag, name)
    return surfaces


def _set_box_transfinite_mesh(
    model,
    volume: int,
    surfaces: list[int],
    *,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    curves: set[int] = set()
    for surface in surfaces:
        model.mesh.setTransfiniteSurface(surface)
        for dim, tag in model.getBoundary([(2, surface)], oriented=False):
            if dim == 1:
                curves.add(tag)

    counts = (nx + 1, ny + 1, nz + 1)
    for curve in curves:
        x_min, y_min, z_min, x_max, y_max, z_max = model.occ.getBoundingBox(1, curve)
        extents = np.array(
            [
                x_max - x_min,
                y_max - y_min,
                z_max - z_min,
            ]
        )
        axis = int(np.argmax(extents))
        model.mesh.setTransfiniteCurve(curve, counts[axis])

    model.mesh.setTransfiniteVolume(volume)


@register_gmsh_domain("box", dimension=3)
def build_box_gmsh_model(model, domain: DomainConfig) -> None:
    """Populate the active Gmsh model with a tagged axis-aligned box."""
    validate_domain_params(domain.type, domain.params)
    Lx, Ly, Lz, nx, ny, nz = _size_and_resolution(domain)

    volume = model.occ.addBox(0.0, 0.0, 0.0, Lx, Ly, Lz)
    model.occ.synchronize()

    model.addPhysicalGroup(3, [volume], tag=1)
    model.setPhysicalName(3, 1, "volume")
    surfaces = _add_box_boundaries(model, volume, Lx=Lx, Ly=Ly, Lz=Lz)
    _set_box_transfinite_mesh(model, volume, surfaces, nx=nx, ny=ny, nz=nz)


@register_domain("box")
def create_box(domain: DomainConfig) -> DomainGeometry:
    """Create a tagged 3D box through Gmsh."""
    Lx, Ly, Lz, _, _, _ = _size_and_resolution(domain)
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
            offset=(-Lx, 0.0, 0.0),
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
            offset=(0.0, -Ly, 0.0),
        ),
        builtin_periodic_map(
            name="z",
            group_id="z",
            slave_boundary="z+",
            master_boundary="z-",
            slave_selector=lambda x, tol, lim=Lz: np.isclose(
                x[2],
                lim,
                atol=tol,
                rtol=tol,
            ),
            offset=(0.0, 0.0, -Lz),
        ),
    )
    domain_geom.periodic_maps = merge_periodic_maps(
        builtin_maps,
        compile_config_periodic_maps(domain.periodic_maps),
    )
    return domain_geom

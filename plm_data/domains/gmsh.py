"""Shared Gmsh model import and tagging helpers for domain packages."""

from typing import Any

import numpy as np
import ufl
from dolfinx import mesh
from dolfinx.mesh import GhostMode
from mpi4py import MPI

from plm_data.domains.base import (
    DomainGeometry,
    build_gmsh_domain_model,
    get_gmsh_domain_dimension,
    register_domain,
    register_gmsh_domain,
)


def model_to_mesh_shared_facet(
    model: Any,
    comm: MPI.Intracomm,
    *,
    rank: int,
    gdim: int,
) -> Any:
    """Import a Gmsh model using shared-facet ghosting for parallel runs."""
    from dolfinx.io.gmsh import model_to_mesh

    partitioner = mesh.create_cell_partitioner(GhostMode.shared_facet, 2)
    return model_to_mesh(model, comm, rank=rank, gdim=gdim, partitioner=partitioner)


def domain_geometry_from_gmsh_mesh_data(mesh_data: Any) -> DomainGeometry:
    """Convert DOLFINx Gmsh import data into the shared domain container."""
    msh = mesh_data.mesh
    boundary_names = {
        name: physical_group.tag
        for name, physical_group in mesh_data.physical_groups.items()
        if physical_group.dim == msh.topology.dim - 1
    }
    ft = mesh_data.facet_tags
    if ft is None:
        raise AssertionError("Gmsh model produced no facet tags for the domain.")
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    return DomainGeometry(
        mesh=msh,
        facet_tags=ft,
        boundary_names=boundary_names,
        ds=ds,
    )


def finalize_gmsh_domain(
    model: Any,
    *,
    name: str,
    gdim: int,
) -> DomainGeometry:
    """Import one fully tagged Gmsh model and wrap it as DomainGeometry."""
    mesh_data = model_to_mesh_shared_facet(model, MPI.COMM_WORLD, rank=0, gdim=gdim)
    domain_geom = domain_geometry_from_gmsh_mesh_data(mesh_data)
    if not domain_geom.boundary_names:
        raise AssertionError(
            f"Gmsh model '{name}' produced no named boundary physical groups."
        )
    return domain_geom


def create_gmsh_domain(domain: Any) -> DomainGeometry:
    """Build, mesh, import, and wrap one registered Gmsh domain."""
    import gmsh

    dimension = get_gmsh_domain_dimension(domain.type)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        model = gmsh.model
        model.add(domain.type)
        model.setCurrent(domain.type)

        if MPI.COMM_WORLD.rank == 0:
            build_gmsh_domain_model(domain, model)
            model.mesh.generate(dimension)

        return finalize_gmsh_domain(model, name=domain.type, gdim=dimension)
    finally:
        gmsh.finalize()


def register_gmsh_domain_factory(
    name: str,
    *,
    dimension: int,
) -> Any:
    """Register a Gmsh model builder and the standard Gmsh domain factory."""

    def decorator(builder: Any) -> Any:
        register_gmsh_domain(name, dimension=dimension)(builder)

        @register_domain(name)
        def _create(domain: Any) -> DomainGeometry:
            return create_gmsh_domain(domain)

        _create.__name__ = f"create_{name}"
        return builder

    return decorator


def tag_single_gmsh_boundary(
    model: Any,
    surface_tags: list[int],
    boundary_name: str,
) -> None:
    """Attach one named physical group covering the full outer boundary."""
    boundary = model.getBoundary(
        [(2, surface_tag) for surface_tag in surface_tags],
        oriented=False,
    )
    boundary_curves = [tag for dim, tag in boundary if dim == 1]
    if not boundary_curves:
        raise AssertionError("Gmsh surface produced no boundary curves.")
    model.addPhysicalGroup(1, boundary_curves, tag=1)
    model.setPhysicalName(1, 1, boundary_name)


def add_named_gmsh_boundaries(
    model: Any,
    boundary_groups: tuple[tuple[str, list[int]], ...],
    *,
    domain_name: str,
) -> None:
    """Attach one named physical group per boundary subset."""
    for physical_tag, (name, curve_tags) in enumerate(boundary_groups, start=1):
        if not curve_tags:
            raise AssertionError(
                f"{domain_name} domain produced no curves for '{name}'."
            )
        model.addPhysicalGroup(1, curve_tags, tag=physical_tag)
        model.setPhysicalName(1, physical_tag, name)


def classify_rectangular_channel_hole_boundaries(
    model: Any,
    surface_tags: list[int],
    *,
    length: float,
    height: float,
    tol: float,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Split channel-with-hole curves into inlet, outlet, walls, and interior."""
    inlet: list[int] = []
    outlet: list[int] = []
    walls: list[int] = []
    interior: list[int] = []
    boundary = model.getBoundary(
        [(2, surface_tag) for surface_tag in surface_tags],
        oriented=False,
    )
    for dim, tag in boundary:
        if dim != 1:
            continue
        bb = model.occ.getBoundingBox(dim, tag)
        x_min, y_min, _, x_max, y_max, _ = bb
        if np.isclose(x_min, 0.0, atol=tol) and np.isclose(x_max, 0.0, atol=tol):
            inlet.append(tag)
        elif np.isclose(x_min, length, atol=tol) and np.isclose(
            x_max,
            length,
            atol=tol,
        ):
            outlet.append(tag)
        elif np.isclose(y_min, 0.0, atol=tol) or np.isclose(y_max, height, atol=tol):
            walls.append(tag)
        else:
            interior.append(tag)
    return inlet, outlet, walls, interior


def fuse_planar_surfaces(model: Any, surface_tags: list[int]) -> list[int]:
    """Fuse 2D OCC surfaces and return the resulting surface tags."""
    if not surface_tags:
        raise AssertionError("Expected at least one surface to fuse.")
    if len(surface_tags) == 1:
        model.occ.synchronize()
        return surface_tags

    model.occ.fuse(
        [(2, surface_tags[0])],
        [(2, surface_tag) for surface_tag in surface_tags[1:]],
        removeObject=True,
        removeTool=True,
    )
    model.occ.synchronize()
    fused_surfaces = [entity[1] for entity in model.getEntities(2)]
    if not fused_surfaces:
        raise AssertionError("OCC fuse produced no 2D surfaces.")
    return fused_surfaces


def add_orthogonal_channel_surfaces(
    model: Any,
    *,
    centerline_points: list[np.ndarray],
    channel_width: float,
) -> list[int]:
    """Create one rounded orthogonal channel by thickening a polyline."""
    half_width = 0.5 * channel_width
    surfaces: list[int] = []

    for start, end in zip(centerline_points, centerline_points[1:]):
        if np.isclose(start[1], end[1]):
            x0 = min(float(start[0]), float(end[0]))
            length = abs(float(end[0] - start[0]))
            surfaces.append(
                model.occ.addRectangle(
                    x0,
                    float(start[1] - half_width),
                    0.0,
                    length,
                    channel_width,
                )
            )
            continue
        if np.isclose(start[0], end[0]):
            y0 = min(float(start[1]), float(end[1]))
            height = abs(float(end[1] - start[1]))
            surfaces.append(
                model.occ.addRectangle(
                    float(start[0] - half_width),
                    y0,
                    0.0,
                    channel_width,
                    height,
                )
            )
            continue
        raise AssertionError(
            "Orthogonal channel centerlines must use axis-aligned segments."
        )

    for point in centerline_points[1:-1]:
        surfaces.append(
            model.occ.addDisk(
                float(point[0]),
                float(point[1]),
                0.0,
                half_width,
                half_width,
            )
        )

    return fuse_planar_surfaces(model, surfaces)


def porous_channel_obstacle_centers(
    *,
    obstacle_radius: float,
    n_rows: int,
    n_cols: int,
    pitch_x: float,
    pitch_y: float,
    x_margin: float,
    y_margin: float,
    row_shift_fraction: float,
) -> list[tuple[float, float]]:
    """Return circle centers for one staggered porous obstacle array."""
    base_x = x_margin + obstacle_radius
    base_y = y_margin + obstacle_radius
    shift_x = row_shift_fraction * pitch_x
    centers: list[tuple[float, float]] = []
    for row in range(n_rows):
        row_shift_x = shift_x if row % 2 == 1 else 0.0
        y = base_y + row * pitch_y
        for col in range(n_cols):
            x = base_x + row_shift_x + col * pitch_x
            centers.append((x, y))
    return centers

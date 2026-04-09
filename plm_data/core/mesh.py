"""Mesh creation helpers with a pluggable domain registry."""

from collections.abc import Callable
from dataclasses import dataclass, field
import math

import numpy as np
import ufl
from dolfinx import mesh
from dolfinx.mesh import CellType, GhostMode, locate_entities_boundary, meshtags
from mpi4py import MPI

from plm_data.core.airfoil import symmetric_naca_airfoil_surfaces
from plm_data.core.config import (
    DomainConfig,
    PeriodicMapConfig,
    validate_domain_params,
)


@dataclass
class PeriodicBoundaryMap:
    """Resolved geometric map for one named periodic boundary pair."""

    name: str
    slave_boundary: str
    master_boundary: str
    matrix: np.ndarray
    offset: np.ndarray
    group_id: str
    slave_selector: Callable[[np.ndarray, float], np.ndarray] | None = None

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Map slave-side coordinates to master-side coordinates."""
        gdim = self.matrix.shape[0]
        out = x.copy()
        out[:gdim, :] = self.matrix @ x[:gdim, :] + self.offset[:, None]
        return out

    def on_slave(self, x: np.ndarray, tol: float) -> np.ndarray:
        """Return a mask selecting points on the slave side."""
        if self.slave_selector is None:
            return np.ones(x.shape[1], dtype=bool)
        return self.slave_selector(x, tol)


@dataclass
class DomainGeometry:
    """Mesh plus named boundary identification."""

    mesh: mesh.Mesh
    facet_tags: mesh.MeshTags  # type: ignore[reportInvalidTypeForm]
    boundary_names: dict[str, int]
    ds: ufl.Measure  # type: ignore[reportInvalidTypeForm]
    periodic_maps: dict[frozenset[str], PeriodicBoundaryMap] = field(
        default_factory=dict
    )

    @property
    def has_periodic_maps(self) -> bool:
        """Return whether the domain exposes any periodic pair maps."""
        return bool(self.periodic_maps)

    def periodic_map(self, side_a: str, side_b: str) -> PeriodicBoundaryMap:
        """Return the resolved periodic map for one side pair."""
        key = frozenset({side_a, side_b})
        if key not in self.periodic_maps:
            raise KeyError(
                f"Domain does not define a periodic map for boundary pair "
                f"{sorted(key)}."
            )
        return self.periodic_maps[key]


DomainFactory = Callable[[DomainConfig], DomainGeometry]
_DOMAIN_REGISTRY: dict[str, DomainFactory] = {}


def register_domain(name: str) -> Callable[[DomainFactory], DomainFactory]:
    """Register a domain factory under a config-facing type name."""

    def decorator(factory: DomainFactory) -> DomainFactory:
        _DOMAIN_REGISTRY[name] = factory
        return factory

    return decorator


def list_domains() -> list[str]:
    """Return the registered domain types."""
    return sorted(_DOMAIN_REGISTRY)


def _require_param(params: dict, key: str, domain_type: str):
    """Require a domain parameter, raising a clear error if missing."""
    if key not in params:
        raise ValueError(
            f"Missing required parameter '{key}' for domain type '{domain_type}'. "
            f"Got params: {params}"
        )
    return params[key]


def create_domain(domain: DomainConfig) -> DomainGeometry:
    """Create a mesh with tagged boundaries from a domain configuration."""
    if domain.type not in _DOMAIN_REGISTRY:
        raise ValueError(
            f"Unknown domain type: '{domain.type}'. "
            f"Available: {', '.join(list_domains())}"
        )
    return _DOMAIN_REGISTRY[domain.type](domain)


def _tag_boundaries(
    msh: mesh.Mesh, predicates: dict[str, Callable]
) -> tuple[mesh.MeshTags, dict[str, int], ufl.Measure]:
    """Tag boundary facets using geometric predicates."""
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)

    boundary_names: dict[str, int] = {}
    all_facets = []
    all_tags = []

    for tag_idx, (name, marker) in enumerate(predicates.items(), start=1):
        boundary_names[name] = tag_idx
        facets = locate_entities_boundary(msh, fdim, marker)
        all_facets.append(facets)
        all_tags.append(np.full_like(facets, tag_idx))

    if all_facets:
        facet_indices = np.concatenate(all_facets)
        tag_values = np.concatenate(all_tags)
        sort_order = np.argsort(facet_indices)
        facet_indices = facet_indices[sort_order]
        tag_values = tag_values[sort_order]
    else:
        facet_indices = np.empty(0, dtype=np.int32)
        tag_values = np.empty(0, dtype=np.int32)

    ft = meshtags(msh, fdim, facet_indices, tag_values)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    return ft, boundary_names, ds


def _compile_config_periodic_maps(
    periodic_maps: dict[str, PeriodicMapConfig],
) -> dict[frozenset[str], PeriodicBoundaryMap]:
    """Compile declarative domain periodic maps into runtime maps."""
    compiled: dict[frozenset[str], PeriodicBoundaryMap] = {}
    for name, map_config in periodic_maps.items():
        key = frozenset({map_config.slave, map_config.master})
        if key in compiled:
            raise ValueError(
                f"Duplicate periodic domain map for boundary pair {sorted(key)}."
            )
        compiled[key] = PeriodicBoundaryMap(
            name=name,
            slave_boundary=map_config.slave,
            master_boundary=map_config.master,
            matrix=np.asarray(map_config.matrix, dtype=float),
            offset=np.asarray(map_config.offset, dtype=float),
            group_id=name,
        )
    return compiled


def _merge_periodic_maps(
    *maps: dict[frozenset[str], PeriodicBoundaryMap],
) -> dict[frozenset[str], PeriodicBoundaryMap]:
    """Merge periodic map dictionaries, rejecting duplicate side pairs."""
    merged: dict[frozenset[str], PeriodicBoundaryMap] = {}
    for periodic_maps in maps:
        for key, periodic_map in periodic_maps.items():
            if key in merged:
                raise ValueError(
                    f"Duplicate periodic map for boundary pair {sorted(key)}."
                )
            merged[key] = periodic_map
    return merged


def _builtin_periodic_map(
    *,
    name: str,
    group_id: str,
    slave_boundary: str,
    master_boundary: str,
    slave_selector: Callable[[np.ndarray, float], np.ndarray],
    offset: tuple[float, ...],
) -> dict[frozenset[str], PeriodicBoundaryMap]:
    """Build one built-in affine periodic pair map."""
    gdim = len(offset)
    return {
        frozenset({slave_boundary, master_boundary}): PeriodicBoundaryMap(
            name=name,
            slave_boundary=slave_boundary,
            master_boundary=master_boundary,
            matrix=np.eye(gdim, dtype=float),
            offset=np.asarray(offset, dtype=float),
            group_id=group_id,
            slave_selector=slave_selector,
        )
    }


def _model_to_mesh_shared_facet(model, comm: MPI.Intracomm, *, rank: int, gdim: int):
    """Import a Gmsh model using shared-facet ghosting for parallel runs."""
    from dolfinx.io.gmsh import model_to_mesh

    partitioner = mesh.create_cell_partitioner(GhostMode.shared_facet, 2)
    return model_to_mesh(model, comm, rank=rank, gdim=gdim, partitioner=partitioner)


def _domain_geometry_from_gmsh_mesh_data(mesh_data) -> DomainGeometry:
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


def _finalize_gmsh_domain(
    model,
    *,
    name: str,
    gdim: int,
) -> DomainGeometry:
    """Import one fully tagged Gmsh model and wrap it as DomainGeometry."""
    comm = MPI.COMM_WORLD
    mesh_data = _model_to_mesh_shared_facet(model, comm, rank=0, gdim=gdim)
    domain_geom = _domain_geometry_from_gmsh_mesh_data(mesh_data)
    if not domain_geom.boundary_names:
        raise AssertionError(
            f"Gmsh model '{name}' produced no named boundary physical groups."
        )
    return domain_geom


def _tag_single_gmsh_boundary(
    model, surface_tags: list[int], boundary_name: str
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


def _fuse_planar_surfaces(model, surface_tags: list[int]) -> list[int]:
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


def _add_orthogonal_channel_surfaces(
    model,
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

    return _fuse_planar_surfaces(model, surfaces)


def _local_coordinates(
    x: np.ndarray,
    origin: np.ndarray,
    inverse_basis: np.ndarray,
) -> np.ndarray:
    """Return local parallelogram coordinates for physical points."""
    return inverse_basis @ (x[:2, :] - origin[:, None])


@register_domain("interval")
def _create_interval(domain: DomainConfig) -> DomainGeometry:
    p = domain.params
    size = _require_param(p, "size", domain.type)
    res = _require_param(p, "mesh_resolution", domain.type)
    length = float(size[0] if isinstance(size, list) else size)
    nx = int(res[0] if isinstance(res, list) else res)

    msh = mesh.create_interval(
        comm=MPI.COMM_WORLD,
        nx=nx,
        points=[0.0, length],
        ghost_mode=GhostMode.shared_facet,
    )

    predicates = {
        "x-": lambda x, lim=0.0: np.isclose(x[0], lim),
        "x+": lambda x, lim=length: np.isclose(x[0], lim),
    }
    ft, boundary_names, ds = _tag_boundaries(msh, predicates)
    builtin_maps = _builtin_periodic_map(
        name="x",
        group_id="x",
        slave_boundary="x+",
        master_boundary="x-",
        slave_selector=lambda x, tol, lim=length: np.isclose(
            x[0], lim, atol=tol, rtol=tol
        ),
        offset=(-length,),
    )
    periodic_maps = _merge_periodic_maps(
        builtin_maps,
        _compile_config_periodic_maps(domain.periodic_maps),
    )
    return DomainGeometry(
        mesh=msh,
        facet_tags=ft,
        boundary_names=boundary_names,
        ds=ds,
        periodic_maps=periodic_maps,
    )


@register_domain("rectangle")
def _create_rectangle(domain: DomainConfig) -> DomainGeometry:
    p = domain.params
    size = _require_param(p, "size", domain.type)
    res = _require_param(p, "mesh_resolution", domain.type)
    Lx, Ly = size[0], size[1]

    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (Lx, Ly)),
        n=(res[0], res[1]),
        cell_type=CellType.triangle,
        ghost_mode=GhostMode.shared_facet,
    )

    predicates = {
        "x-": lambda x, lim=0.0: np.isclose(x[0], lim),
        "x+": lambda x, lim=Lx: np.isclose(x[0], lim),
        "y-": lambda x, lim=0.0: np.isclose(x[1], lim),
        "y+": lambda x, lim=Ly: np.isclose(x[1], lim),
    }
    ft, boundary_names, ds = _tag_boundaries(msh, predicates)
    builtin_maps = _merge_periodic_maps(
        _builtin_periodic_map(
            name="x",
            group_id="x",
            slave_boundary="x+",
            master_boundary="x-",
            slave_selector=lambda x, tol, lim=Lx: np.isclose(
                x[0], lim, atol=tol, rtol=tol
            ),
            offset=(-Lx, 0.0),
        ),
        _builtin_periodic_map(
            name="y",
            group_id="y",
            slave_boundary="y+",
            master_boundary="y-",
            slave_selector=lambda x, tol, lim=Ly: np.isclose(
                x[1], lim, atol=tol, rtol=tol
            ),
            offset=(0.0, -Ly),
        ),
    )
    periodic_maps = _merge_periodic_maps(
        builtin_maps,
        _compile_config_periodic_maps(domain.periodic_maps),
    )
    return DomainGeometry(
        mesh=msh,
        facet_tags=ft,
        boundary_names=boundary_names,
        ds=ds,
        periodic_maps=periodic_maps,
    )


@register_domain("box")
def _create_box(domain: DomainConfig) -> DomainGeometry:
    p = domain.params
    size = _require_param(p, "size", domain.type)
    res = _require_param(p, "mesh_resolution", domain.type)
    Lx, Ly, Lz = size[0], size[1], size[2]

    msh = mesh.create_box(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0, 0.0), (Lx, Ly, Lz)),  # type: ignore[reportArgumentType]
        n=(res[0], res[1], res[2]),
        cell_type=CellType.tetrahedron,
        ghost_mode=GhostMode.shared_facet,
    )

    predicates = {
        "x-": lambda x, lim=0.0: np.isclose(x[0], lim),
        "x+": lambda x, lim=Lx: np.isclose(x[0], lim),
        "y-": lambda x, lim=0.0: np.isclose(x[1], lim),
        "y+": lambda x, lim=Ly: np.isclose(x[1], lim),
        "z-": lambda x, lim=0.0: np.isclose(x[2], lim),
        "z+": lambda x, lim=Lz: np.isclose(x[2], lim),
    }
    ft, boundary_names, ds = _tag_boundaries(msh, predicates)
    builtin_maps = _merge_periodic_maps(
        _builtin_periodic_map(
            name="x",
            group_id="x",
            slave_boundary="x+",
            master_boundary="x-",
            slave_selector=lambda x, tol, lim=Lx: np.isclose(
                x[0], lim, atol=tol, rtol=tol
            ),
            offset=(-Lx, 0.0, 0.0),
        ),
        _builtin_periodic_map(
            name="y",
            group_id="y",
            slave_boundary="y+",
            master_boundary="y-",
            slave_selector=lambda x, tol, lim=Ly: np.isclose(
                x[1], lim, atol=tol, rtol=tol
            ),
            offset=(0.0, -Ly, 0.0),
        ),
        _builtin_periodic_map(
            name="z",
            group_id="z",
            slave_boundary="z+",
            master_boundary="z-",
            slave_selector=lambda x, tol, lim=Lz: np.isclose(
                x[2], lim, atol=tol, rtol=tol
            ),
            offset=(0.0, 0.0, -Lz),
        ),
    )
    periodic_maps = _merge_periodic_maps(
        builtin_maps,
        _compile_config_periodic_maps(domain.periodic_maps),
    )
    return DomainGeometry(
        mesh=msh,
        facet_tags=ft,
        boundary_names=boundary_names,
        ds=ds,
        periodic_maps=periodic_maps,
    )


@register_domain("disk")
def _create_disk(domain: DomainConfig) -> DomainGeometry:
    import gmsh

    p = domain.params
    center = np.asarray(_require_param(p, "center", domain.type), dtype=float)
    radius = float(_require_param(p, "radius", domain.type))
    mesh_size = float(_require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        model = gmsh.model
        model.add("disk")
        model.setCurrent("disk")

        if MPI.COMM_WORLD.rank == 0:
            disk = model.occ.addDisk(center[0], center[1], 0.0, radius, radius)
            model.occ.synchronize()

            model.addPhysicalGroup(2, [disk], tag=1)
            model.setPhysicalName(2, 1, "surface")
            _tag_single_gmsh_boundary(model, [disk], "outer")

            model.mesh.setSize(model.getEntities(0), mesh_size)
            model.mesh.generate(2)

        return _finalize_gmsh_domain(model, name="disk", gdim=2)
    finally:
        gmsh.finalize()


@register_domain("dumbbell")
def _create_dumbbell(domain: DomainConfig) -> DomainGeometry:
    import gmsh

    p = domain.params
    left_center = np.asarray(_require_param(p, "left_center", domain.type), dtype=float)
    right_center = np.asarray(
        _require_param(p, "right_center", domain.type),
        dtype=float,
    )
    lobe_radius = float(_require_param(p, "lobe_radius", domain.type))
    neck_width = float(_require_param(p, "neck_width", domain.type))
    mesh_size = float(_require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        model = gmsh.model
        model.add("dumbbell")
        model.setCurrent("dumbbell")

        if MPI.COMM_WORLD.rank == 0:
            left_disk = model.occ.addDisk(
                left_center[0],
                left_center[1],
                0.0,
                lobe_radius,
                lobe_radius,
            )
            right_disk = model.occ.addDisk(
                right_center[0],
                right_center[1],
                0.0,
                lobe_radius,
                lobe_radius,
            )
            bridge = model.occ.addRectangle(
                left_center[0],
                left_center[1] - 0.5 * neck_width,
                0.0,
                right_center[0] - left_center[0],
                neck_width,
            )
            model.occ.fuse(
                [(2, left_disk), (2, right_disk)],
                [(2, bridge)],
                removeObject=True,
                removeTool=True,
            )
            model.occ.synchronize()

            surfaces = [entity[1] for entity in model.getEntities(2)]
            model.addPhysicalGroup(2, surfaces, tag=1)
            model.setPhysicalName(2, 1, "surface")
            _tag_single_gmsh_boundary(model, surfaces, "outer")

            model.mesh.setSize(model.getEntities(0), mesh_size)
            model.mesh.generate(2)

        return _finalize_gmsh_domain(model, name="dumbbell", gdim=2)
    finally:
        gmsh.finalize()


@register_domain("l_shape")
def _create_l_shape(domain: DomainConfig) -> DomainGeometry:
    import gmsh

    p = domain.params
    outer_width = float(_require_param(p, "outer_width", domain.type))
    outer_height = float(_require_param(p, "outer_height", domain.type))
    cutout_width = float(_require_param(p, "cutout_width", domain.type))
    cutout_height = float(_require_param(p, "cutout_height", domain.type))
    mesh_size = float(_require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    reentrant_x = outer_width - cutout_width
    reentrant_y = outer_height - cutout_height

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        model = gmsh.model
        model.add("l_shape")
        model.setCurrent("l_shape")

        if MPI.COMM_WORLD.rank == 0:
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
            surfaces = _fuse_planar_surfaces(model, [lower_arm, left_arm])
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
                is_notch_vertical = np.isclose(
                    x_min, reentrant_x, atol=tol
                ) and np.isclose(x_max, reentrant_x, atol=tol)
                is_notch_horizontal = np.isclose(
                    y_min,
                    reentrant_y,
                    atol=tol,
                ) and np.isclose(y_max, reentrant_y, atol=tol)
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
                    raise AssertionError(
                        f"L-shape domain produced no curves for '{name}'."
                    )
                model.addPhysicalGroup(1, curve_tags, tag=physical_tag)
                model.setPhysicalName(1, physical_tag, name)

            model.mesh.setSize(model.getEntities(0), mesh_size)
            model.mesh.generate(2)

        return _finalize_gmsh_domain(model, name="l_shape", gdim=2)
    finally:
        gmsh.finalize()


@register_domain("parallelogram")
def _create_parallelogram(domain: DomainConfig) -> DomainGeometry:
    p = domain.params
    origin = np.asarray(_require_param(p, "origin", domain.type), dtype=float)
    axis_x = np.asarray(_require_param(p, "axis_x", domain.type), dtype=float)
    axis_y = np.asarray(_require_param(p, "axis_y", domain.type), dtype=float)
    res = _require_param(p, "mesh_resolution", domain.type)
    validate_domain_params(domain.type, p)

    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (1.0, 1.0)),
        n=(res[0], res[1]),
        cell_type=CellType.triangle,
        ghost_mode=GhostMode.shared_facet,
    )

    predicates = {
        "x-": lambda x: np.isclose(x[0], 0.0),
        "x+": lambda x: np.isclose(x[0], 1.0),
        "y-": lambda x: np.isclose(x[1], 0.0),
        "y+": lambda x: np.isclose(x[1], 1.0),
    }
    ft, boundary_names, ds = _tag_boundaries(msh, predicates)

    reference_coords = msh.geometry.x[:, :2].copy()
    transformed_coords = (
        origin
        + reference_coords[:, [0]] * axis_x[None, :]
        + reference_coords[:, [1]] * axis_y[None, :]
    )
    msh.geometry.x[:, :2] = transformed_coords

    inverse_basis = np.linalg.inv(np.column_stack((axis_x, axis_y)))
    builtin_maps = _merge_periodic_maps(
        _builtin_periodic_map(
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
        _builtin_periodic_map(
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
    periodic_maps = _merge_periodic_maps(
        builtin_maps,
        _compile_config_periodic_maps(domain.periodic_maps),
    )
    return DomainGeometry(
        mesh=msh,
        facet_tags=ft,
        boundary_names=boundary_names,
        ds=ds,
        periodic_maps=periodic_maps,
    )


@register_domain("channel_obstacle")
def _create_channel_obstacle(domain: DomainConfig) -> DomainGeometry:
    import gmsh

    p = domain.params
    length = float(_require_param(p, "length", domain.type))
    height = float(_require_param(p, "height", domain.type))
    obstacle_center = np.asarray(
        _require_param(p, "obstacle_center", domain.type),
        dtype=float,
    )
    obstacle_radius = float(_require_param(p, "obstacle_radius", domain.type))
    mesh_size = float(_require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        model = gmsh.model
        model.add("channel_obstacle")
        model.setCurrent("channel_obstacle")

        if MPI.COMM_WORLD.rank == 0:
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
            inlet: list[int] = []
            outlet: list[int] = []
            walls: list[int] = []
            obstacle_curves: list[int] = []
            boundary = model.getBoundary(
                [(2, surface_tag) for surface_tag in surfaces],
                oriented=False,
            )
            for dim, tag in boundary:
                if dim != 1:
                    continue
                bb = model.occ.getBoundingBox(dim, tag)
                x_min, y_min, _, x_max, y_max, _ = bb
                if np.isclose(x_min, 0.0, atol=tol) and np.isclose(
                    x_max,
                    0.0,
                    atol=tol,
                ):
                    inlet.append(tag)
                elif np.isclose(x_min, length, atol=tol) and np.isclose(
                    x_max,
                    length,
                    atol=tol,
                ):
                    outlet.append(tag)
                elif np.isclose(y_min, 0.0, atol=tol) or np.isclose(
                    y_max,
                    height,
                    atol=tol,
                ):
                    walls.append(tag)
                else:
                    obstacle_curves.append(tag)

            for physical_tag, (name, curve_tags) in enumerate(
                (
                    ("inlet", inlet),
                    ("outlet", outlet),
                    ("walls", walls),
                    ("obstacle", obstacle_curves),
                ),
                start=1,
            ):
                if not curve_tags:
                    raise AssertionError(
                        f"Channel-obstacle domain produced no curves for '{name}'."
                    )
                model.addPhysicalGroup(1, curve_tags, tag=physical_tag)
                model.setPhysicalName(1, physical_tag, name)

            model.mesh.setSize(model.getEntities(0), mesh_size)
            model.mesh.generate(2)

        return _finalize_gmsh_domain(model, name="channel_obstacle", gdim=2)
    finally:
        gmsh.finalize()


@register_domain("annulus")
def _create_annulus(domain: DomainConfig) -> DomainGeometry:
    import gmsh

    p = domain.params
    center = np.asarray(_require_param(p, "center", domain.type), dtype=float)
    inner_radius = float(_require_param(p, "inner_radius", domain.type))
    outer_radius = float(_require_param(p, "outer_radius", domain.type))
    mesh_size = float(_require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    comm = MPI.COMM_WORLD
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        model = gmsh.model
        model.add("annulus")
        model.setCurrent("annulus")

        if comm.rank == 0:
            outer_disk = model.occ.addDisk(
                center[0],
                center[1],
                0.0,
                outer_radius,
                outer_radius,
            )
            inner_disk = model.occ.addDisk(
                center[0],
                center[1],
                0.0,
                inner_radius,
                inner_radius,
            )
            model.occ.cut([(2, outer_disk)], [(2, inner_disk)])
            model.occ.synchronize()

            surfaces = [e[1] for e in model.getEntities(2)]
            model.addPhysicalGroup(2, surfaces, tag=1)
            model.setPhysicalName(2, 1, "surface")

            boundary = model.getBoundary([(2, s) for s in surfaces], oriented=False)
            r_mid = inner_radius + outer_radius
            inner_curves: list[int] = []
            outer_curves: list[int] = []
            for dim, tag in boundary:
                bb = model.occ.getBoundingBox(dim, tag)
                extent = max(bb[3] - bb[0], bb[4] - bb[1])
                if extent < r_mid:
                    inner_curves.append(tag)
                else:
                    outer_curves.append(tag)

            model.addPhysicalGroup(1, inner_curves, tag=1)
            model.setPhysicalName(1, 1, "inner")
            model.addPhysicalGroup(1, outer_curves, tag=2)
            model.setPhysicalName(1, 2, "outer")

            model.mesh.setSize(model.getEntities(0), mesh_size)
            model.mesh.generate(2)

        mesh_data = _model_to_mesh_shared_facet(model, comm, rank=0, gdim=2)
    finally:
        gmsh.finalize()
    return _domain_geometry_from_gmsh_mesh_data(mesh_data)


@register_domain("y_bifurcation")
def _create_y_bifurcation(domain: DomainConfig) -> DomainGeometry:
    import gmsh

    p = domain.params
    inlet_length = float(_require_param(p, "inlet_length", domain.type))
    branch_length = float(_require_param(p, "branch_length", domain.type))
    branch_angle_degrees = float(_require_param(p, "branch_angle_degrees", domain.type))
    channel_width = float(_require_param(p, "channel_width", domain.type))
    mesh_size = float(_require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    branch_angle = math.radians(branch_angle_degrees)
    branch_axis = np.array([math.cos(branch_angle), math.sin(branch_angle)])
    junction_center = np.array([inlet_length, 0.0])
    inlet_center = np.array([0.0, 0.0])
    outlet_upper_center = junction_center + branch_length * branch_axis
    outlet_lower_center = junction_center + branch_length * np.array(
        [branch_axis[0], -branch_axis[1]]
    )
    cap_tolerance = 0.45 * channel_width + 1.0e-8

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        model = gmsh.model
        model.add("y_bifurcation")
        model.setCurrent("y_bifurcation")

        if MPI.COMM_WORLD.rank == 0:
            half_width = 0.5 * channel_width
            trunk = model.occ.addRectangle(
                0.0,
                -half_width,
                0.0,
                inlet_length,
                channel_width,
            )
            upper_branch = model.occ.addRectangle(
                inlet_length,
                -half_width,
                0.0,
                branch_length,
                channel_width,
            )
            lower_branch = model.occ.addRectangle(
                inlet_length,
                -half_width,
                0.0,
                branch_length,
                channel_width,
            )
            model.occ.rotate(
                [(2, upper_branch)],
                inlet_length,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                branch_angle,
            )
            model.occ.rotate(
                [(2, lower_branch)],
                inlet_length,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                -branch_angle,
            )
            junction_disk = model.occ.addDisk(
                inlet_length,
                0.0,
                0.0,
                half_width,
                half_width,
            )
            model.occ.fuse(
                [(2, trunk)],
                [
                    (2, upper_branch),
                    (2, lower_branch),
                    (2, junction_disk),
                ],
                removeObject=True,
                removeTool=True,
            )
            model.occ.synchronize()

            surfaces = [entity[1] for entity in model.getEntities(2)]
            model.addPhysicalGroup(2, surfaces, tag=1)
            model.setPhysicalName(2, 1, "surface")

            inlet: list[int] = []
            outlet_upper: list[int] = []
            outlet_lower: list[int] = []
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
                if np.linalg.norm(center - outlet_upper_center) <= cap_tolerance:
                    outlet_upper.append(tag)
                    continue
                if np.linalg.norm(center - outlet_lower_center) <= cap_tolerance:
                    outlet_lower.append(tag)
                    continue
                walls.append(tag)

            for physical_tag, (name, curve_tags) in enumerate(
                (
                    ("inlet", inlet),
                    ("outlet_upper", outlet_upper),
                    ("outlet_lower", outlet_lower),
                    ("walls", walls),
                ),
                start=1,
            ):
                if not curve_tags:
                    raise AssertionError(
                        f"Y-bifurcation domain produced no curves for '{name}'."
                    )
                model.addPhysicalGroup(1, curve_tags, tag=physical_tag)
                model.setPhysicalName(1, physical_tag, name)

            model.mesh.setSize(model.getEntities(0), mesh_size)
            model.mesh.generate(2)

        return _finalize_gmsh_domain(model, name="y_bifurcation", gdim=2)
    finally:
        gmsh.finalize()


@register_domain("venturi_channel")
def _create_venturi_channel(domain: DomainConfig) -> DomainGeometry:
    import gmsh

    p = domain.params
    length = float(_require_param(p, "length", domain.type))
    height = float(_require_param(p, "height", domain.type))
    throat_height = float(_require_param(p, "throat_height", domain.type))
    constriction_center_x = float(
        _require_param(p, "constriction_center_x", domain.type)
    )
    constriction_radius = float(_require_param(p, "constriction_radius", domain.type))
    mesh_size = float(_require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    indentation = 0.5 * (height - throat_height)
    top_center_y = height + constriction_radius - indentation
    bottom_center_y = -constriction_radius + indentation

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        model = gmsh.model
        model.add("venturi_channel")
        model.setCurrent("venturi_channel")

        if MPI.COMM_WORLD.rank == 0:
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
            model.mesh.generate(2)

        return _finalize_gmsh_domain(model, name="venturi_channel", gdim=2)
    finally:
        gmsh.finalize()


@register_domain("serpentine_channel")
def _create_serpentine_channel(domain: DomainConfig) -> DomainGeometry:
    import gmsh

    p = domain.params
    channel_length = float(_require_param(p, "channel_length", domain.type))
    lane_spacing = float(_require_param(p, "lane_spacing", domain.type))
    n_bends = int(_require_param(p, "n_bends", domain.type))
    channel_width = float(_require_param(p, "channel_width", domain.type))
    mesh_size = float(_require_param(p, "mesh_size", domain.type))
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

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        model = gmsh.model
        model.add("serpentine_channel")
        model.setCurrent("serpentine_channel")

        if MPI.COMM_WORLD.rank == 0:
            surfaces = _add_orthogonal_channel_surfaces(
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
            model.mesh.generate(2)

        return _finalize_gmsh_domain(model, name="serpentine_channel", gdim=2)
    finally:
        gmsh.finalize()


@register_domain("airfoil_channel")
def _create_airfoil_channel(domain: DomainConfig) -> DomainGeometry:
    import gmsh

    p = domain.params
    length = float(_require_param(p, "length", domain.type))
    height = float(_require_param(p, "height", domain.type))
    airfoil_center = np.asarray(
        _require_param(p, "airfoil_center", domain.type),
        dtype=float,
    )
    chord_length = float(_require_param(p, "chord_length", domain.type))
    thickness_ratio = float(_require_param(p, "thickness_ratio", domain.type))
    attack_angle_degrees = float(_require_param(p, "attack_angle_degrees", domain.type))
    mesh_size = float(_require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    upper_points, lower_points = symmetric_naca_airfoil_surfaces(
        chord_length=chord_length,
        thickness_ratio=thickness_ratio,
        center=airfoil_center,
        attack_angle_degrees=attack_angle_degrees,
    )

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        model = gmsh.model
        model.add("airfoil_channel")
        model.setCurrent("airfoil_channel")

        if MPI.COMM_WORLD.rank == 0:
            channel = model.occ.addRectangle(0.0, 0.0, 0.0, length, height)

            upper_tags = [
                model.occ.addPoint(float(x_coord), float(y_coord), 0.0)
                for x_coord, y_coord in upper_points
            ]
            leading_edge_tag = upper_tags[-1]
            lower_tags = [leading_edge_tag] + [
                model.occ.addPoint(float(x_coord), float(y_coord), 0.0)
                for x_coord, y_coord in lower_points
            ]
            upper_spline = model.occ.addSpline(upper_tags)
            lower_spline = model.occ.addSpline(lower_tags)
            trailing_edge = model.occ.addLine(lower_tags[-1], upper_tags[0])
            airfoil_loop = model.occ.addCurveLoop(
                [upper_spline, lower_spline, trailing_edge]
            )
            airfoil_surface = model.occ.addPlaneSurface([airfoil_loop])

            model.occ.cut(
                [(2, channel)],
                [(2, airfoil_surface)],
                removeObject=True,
                removeTool=True,
            )
            model.occ.synchronize()

            surfaces = [entity[1] for entity in model.getEntities(2)]
            model.addPhysicalGroup(2, surfaces, tag=1)
            model.setPhysicalName(2, 1, "surface")

            tol = max(mesh_size, 1.0e-8)
            inlet: list[int] = []
            outlet: list[int] = []
            walls: list[int] = []
            airfoil_curves: list[int] = []
            boundary = model.getBoundary(
                [(2, surface_tag) for surface_tag in surfaces],
                oriented=False,
            )
            for dim, tag in boundary:
                if dim != 1:
                    continue
                bb = model.occ.getBoundingBox(dim, tag)
                x_min, y_min, _, x_max, y_max, _ = bb
                if np.isclose(x_min, 0.0, atol=tol) and np.isclose(
                    x_max,
                    0.0,
                    atol=tol,
                ):
                    inlet.append(tag)
                elif np.isclose(x_min, length, atol=tol) and np.isclose(
                    x_max,
                    length,
                    atol=tol,
                ):
                    outlet.append(tag)
                elif np.isclose(y_min, 0.0, atol=tol) or np.isclose(
                    y_max,
                    height,
                    atol=tol,
                ):
                    walls.append(tag)
                else:
                    airfoil_curves.append(tag)

            for physical_tag, (name, curve_tags) in enumerate(
                (
                    ("inlet", inlet),
                    ("outlet", outlet),
                    ("walls", walls),
                    ("airfoil", airfoil_curves),
                ),
                start=1,
            ):
                if not curve_tags:
                    raise AssertionError(
                        f"Airfoil-channel domain produced no curves for '{name}'."
                    )
                model.addPhysicalGroup(1, curve_tags, tag=physical_tag)
                model.setPhysicalName(1, physical_tag, name)

            model.mesh.setSize(model.getEntities(0), mesh_size)
            model.mesh.generate(2)

        return _finalize_gmsh_domain(model, name="airfoil_channel", gdim=2)
    finally:
        gmsh.finalize()

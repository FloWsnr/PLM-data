"""Mesh creation helpers with a pluggable domain registry."""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import ufl
from dolfinx import mesh
from dolfinx.mesh import CellType, GhostMode, locate_entities_boundary, meshtags
from mpi4py import MPI

from plm_data.core.config import DomainConfig, PeriodicMapConfig


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


@register_domain("annulus")
def _create_annulus(domain: DomainConfig) -> DomainGeometry:
    import gmsh
    from dolfinx.io.gmsh import model_to_mesh

    p = domain.params
    inner_radius = float(_require_param(p, "inner_radius", domain.type))
    outer_radius = float(_require_param(p, "outer_radius", domain.type))
    mesh_size = float(_require_param(p, "mesh_size", domain.type))

    comm = MPI.COMM_WORLD
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        model = gmsh.model
        model.add("annulus")
        model.setCurrent("annulus")

        if comm.rank == 0:
            outer_disk = model.occ.addDisk(0.0, 0.0, 0.0, outer_radius, outer_radius)
            inner_disk = model.occ.addDisk(0.0, 0.0, 0.0, inner_radius, inner_radius)
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

        mesh_data = model_to_mesh(model, comm, rank=0, gdim=2)
    finally:
        gmsh.finalize()

    msh = mesh_data.mesh
    boundary_names: dict[str, int] = {}
    for name, pg in mesh_data.physical_groups.items():
        if pg.dim == 1:
            boundary_names[name] = pg.tag

    ft = mesh_data.facet_tags
    assert ft is not None, "Gmsh model produced no facet tags for the annulus."
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)

    return DomainGeometry(
        mesh=msh,
        facet_tags=ft,
        boundary_names=boundary_names,
        ds=ds,
    )

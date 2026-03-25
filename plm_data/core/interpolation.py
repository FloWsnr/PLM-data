"""Interpolate DOLFINx functions onto regular grids for data output.

Uses non-matching mesh interpolation (C++-backed) for performance:
creates a structured output mesh once, precomputes interpolation data,
then reuses it for all subsequent frames.
"""

from dataclasses import dataclass, field

import numpy as np
from dolfinx import fem, mesh as dmesh
from dolfinx.mesh import CellType, GhostMode
from mpi4py import MPI


@dataclass
class InterpolationCache:
    """Cached interpolation infrastructure for a given source→output mesh pair.

    Created once per (source function space, output resolution) combination.
    Reused across all frames to avoid recomputing the expensive
    create_interpolation_data step.
    """

    out_mesh: dmesh.Mesh
    V_out: fem.FunctionSpace
    u_out: fem.Function
    cells: np.ndarray
    interp_data: dict = field(default_factory=dict)
    # Grid reordering indices
    grid_indices: list[np.ndarray] = field(default_factory=list)
    resolution: tuple[int, ...] = ()
    gdim: int = 0


def _create_cache(
    V_from: fem.FunctionSpace,
    resolution: tuple[int, ...],
    bounds: tuple[tuple[float, float], ...],
) -> InterpolationCache:
    """Build the output mesh, function space, and interpolation data."""
    gdim = V_from.mesh.geometry.dim

    if gdim == 2:
        out_mesh = dmesh.create_rectangle(
            comm=MPI.COMM_WORLD,
            points=((bounds[0][0], bounds[1][0]), (bounds[0][1], bounds[1][1])),
            n=(resolution[0] - 1, resolution[1] - 1),
            cell_type=CellType.triangle,
            ghost_mode=GhostMode.shared_facet,
        )
    elif gdim == 3:
        out_mesh = dmesh.create_box(
            comm=MPI.COMM_WORLD,
            points=(  # type: ignore[reportArgumentType]
                (bounds[0][0], bounds[1][0], bounds[2][0]),
                (bounds[0][1], bounds[1][1], bounds[2][1]),
            ),
            n=(resolution[0] - 1, resolution[1] - 1, resolution[2] - 1),
            cell_type=CellType.tetrahedron,
            ghost_mode=GhostMode.shared_facet,
        )
    else:
        raise ValueError(f"Unsupported geometry dimension: {gdim}")

    V_out = fem.functionspace(out_mesh, ("Lagrange", 1))
    u_out = fem.Function(V_out)

    cell_map = out_mesh.topology.index_map(out_mesh.topology.dim)
    num_cells = cell_map.size_local + cell_map.num_ghosts
    cells = np.arange(num_cells, dtype=np.int32)

    interp_data = fem.create_interpolation_data(V_out, V_from, cells, padding=1e-14)

    # Precompute grid reordering indices
    dof_coords = V_out.tabulate_dof_coordinates()
    grid_indices = []
    for d in range(gdim):
        spacing = (bounds[d][1] - bounds[d][0]) / (resolution[d] - 1)
        idx = np.round((dof_coords[:, d] - bounds[d][0]) / spacing).astype(int)
        idx = np.clip(idx, 0, resolution[d] - 1)
        grid_indices.append(idx)

    cache = InterpolationCache(
        out_mesh=out_mesh,
        V_out=V_out,
        u_out=u_out,
        cells=cells,
        grid_indices=grid_indices,
        resolution=resolution,
        gdim=gdim,
    )
    # Store interp_data keyed by source function space id
    cache.interp_data[id(V_from)] = interp_data
    return cache


def function_to_array(
    func: fem.Function,
    resolution: tuple[int, ...],
    bounds: tuple[tuple[float, float], ...] | None = None,
    cache: InterpolationCache | None = None,
) -> tuple[np.ndarray, InterpolationCache]:
    """Interpolate a DOLFINx function onto a regular grid.

    Args:
        func: A DOLFINx Function to evaluate.
        resolution: Grid resolution, e.g. (nx, ny) for 2D or (nx, ny, nz) for 3D.
        bounds: Domain bounds. If None, inferred from the mesh geometry.
        cache: Reusable interpolation cache. If None, one is created.

    Returns:
        Tuple of (array of shape `resolution`, cache for reuse).
    """
    src_mesh = func.function_space.mesh
    gdim = src_mesh.geometry.dim

    if bounds is None:
        coords = src_mesh.geometry.x
        bounds = tuple(
            (float(coords[:, d].min()), float(coords[:, d].max())) for d in range(gdim)
        )

    V_from = func.function_space

    if cache is None:
        cache = _create_cache(V_from, resolution, bounds)

    # If this function space hasn't been seen, compute its interpolation data
    if id(V_from) not in cache.interp_data:
        cache.interp_data[id(V_from)] = fem.create_interpolation_data(
            cache.V_out, V_from, cache.cells, padding=1e-14
        )

    interp_data = cache.interp_data[id(V_from)]
    cache.u_out.interpolate_nonmatching(
        func, cache.cells, interpolation_data=interp_data
    )

    # Reorder DOFs to grid layout
    dof_count = len(cache.grid_indices[0])
    result = np.full(cache.resolution, np.nan)
    if gdim == 2:
        result[cache.grid_indices[1], cache.grid_indices[0]] = cache.u_out.x.array[
            :dof_count
        ]
    elif gdim == 3:
        result[cache.grid_indices[0], cache.grid_indices[1], cache.grid_indices[2]] = (
            cache.u_out.x.array[:dof_count]
        )

    nan_count = int(np.isnan(result).sum())
    if nan_count > 0:
        raise RuntimeError(
            f"Interpolation produced {nan_count}/{result.size} NaN values. "
            "This may indicate solver divergence or points outside the mesh."
        )
    return result, cache

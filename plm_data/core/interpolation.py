"""Interpolate DOLFINx functions onto regular grids for data output.

Uses direct point evaluation via ``Function.eval()`` with a precomputed
owning-cell lookup. In parallel, rank 0 owns the regular output grid
and the ranks cooperatively evaluate the points they own on the
partitioned mesh.
"""

from dataclasses import dataclass

import numpy as np
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.geometry import (
    bb_tree,
    compute_colliding_cells,
    compute_collisions_points,
    determine_point_ownership,
)
from mpi4py import MPI


@dataclass
class InterpolationCache:
    """Cached grid points and cell assignments for direct point evaluation.

    Created once per (source mesh, output resolution) combination.
    Reused across all frames; only `func.eval()` is called per frame.
    """

    points: np.ndarray  # (N_local, 3) evaluation points owned by this rank
    cells: np.ndarray  # (N_local,) local owning cell index per point (-1 = outside)
    resolution: tuple[int, ...]
    gdim: int
    total_points: int
    root_point_indices: tuple[np.ndarray, ...] | None = None
    outside_mask: np.ndarray | None = None  # (total_points,) True for out-of-domain


def _global_bounds(src_mesh: dmesh.Mesh) -> tuple[tuple[float, float], ...]:
    """Compute global mesh bounds across all MPI ranks."""
    coords = src_mesh.geometry.x
    comm = src_mesh.comm
    gdim = src_mesh.geometry.dim
    bounds: list[tuple[float, float]] = []
    for d in range(gdim):
        local_min = float(coords[:, d].min()) if coords.size > 0 else np.inf
        local_max = float(coords[:, d].max()) if coords.size > 0 else -np.inf
        global_min = comm.allreduce(local_min, op=MPI.MIN)
        global_max = comm.allreduce(local_max, op=MPI.MAX)
        bounds.append((global_min, global_max))
    return tuple(bounds)


def _build_grid_points(
    dtype: np.dtype,
    resolution: tuple[int, ...],
    bounds: tuple[tuple[float, float], ...],
) -> np.ndarray:
    """Construct regular-grid evaluation points with DOLFINx's `(N, 3)` layout."""
    axes = [
        np.linspace(bounds[d][0], bounds[d][1], resolution[d])
        for d in range(len(bounds))
    ]
    grids = np.meshgrid(*axes, indexing="ij")
    flat = [g.ravel() for g in grids]

    n_pts = flat[0].size
    points = np.zeros((n_pts, 3), dtype=dtype)
    for d in range(len(bounds)):
        points[:, d] = flat[d]
    return points


def _point_ownership_padding(
    bounds: tuple[tuple[float, float], ...], resolution: tuple[int, ...]
) -> float:
    """Choose a small absolute padding for MPI ownership queries."""
    spans = [upper - lower for lower, upper in bounds]
    max_span = max(spans, default=1.0)
    positive_spacings = [
        span / (resolution[d] - 1)
        for d, span in enumerate(spans)
        if resolution[d] > 1 and span > 0.0
    ]
    min_spacing = min(positive_spacings, default=max_span)
    return max(
        np.finfo(np.float64).eps * max(max_span, 1.0) * 1.0e4, min_spacing * 1.0e-6
    )


def _create_serial_cache(
    src_mesh: dmesh.Mesh,
    resolution: tuple[int, ...],
    bounds: tuple[tuple[float, float], ...],
) -> InterpolationCache:
    """Build point ownership data for a serial mesh."""
    gdim = src_mesh.geometry.dim
    points = _build_grid_points(src_mesh.geometry.x.dtype, resolution, bounds)

    tree = bb_tree(src_mesh, src_mesh.topology.dim)
    candidates = compute_collisions_points(tree, points)
    colliding = compute_colliding_cells(src_mesh, candidates, points)

    cells = np.empty(points.shape[0], dtype=np.int32)
    for i in range(points.shape[0]):
        links = colliding.links(i)
        cells[i] = links[0] if links.size > 0 else -1

    outside = cells < 0
    outside_mask = np.asarray(outside) if outside.any() else None

    return InterpolationCache(
        points=points,
        cells=cells,
        resolution=resolution,
        gdim=gdim,
        total_points=points.shape[0],
        root_point_indices=(np.arange(points.shape[0], dtype=np.int32),),
        outside_mask=outside_mask,
    )


def _create_parallel_cache(
    src_mesh: dmesh.Mesh,
    resolution: tuple[int, ...],
    bounds: tuple[tuple[float, float], ...],
) -> InterpolationCache:
    """Build MPI ownership data for a root-owned regular grid."""
    comm = src_mesh.comm
    rank = comm.rank
    gdim = src_mesh.geometry.dim
    points = _build_grid_points(src_mesh.geometry.x.dtype, resolution, bounds)
    root_points = points.copy() if rank == 0 else np.empty((0, 3), dtype=points.dtype)
    ownership = determine_point_ownership(
        src_mesh,
        root_points,
        _point_ownership_padding(bounds, resolution),
    )

    root_point_indices: tuple[np.ndarray, ...] | None = None
    outside_mask_local: np.ndarray | None = None
    if rank == 0:
        src_owner = np.asarray(ownership.src_owner, dtype=np.int32)
        outside = src_owner < 0
        outside_mask_local = np.asarray(outside) if outside.any() else None
        root_point_indices = tuple(
            np.flatnonzero(src_owner == owner).astype(np.int32)
            for owner in range(comm.size)
        )

    outside_mask_local = comm.bcast(outside_mask_local, root=0)

    return InterpolationCache(
        points=np.asarray(ownership.dest_points, dtype=points.dtype).copy(),
        cells=np.asarray(ownership.dest_cells, dtype=np.int32).copy(),
        resolution=resolution,
        gdim=gdim,
        total_points=points.shape[0],
        root_point_indices=root_point_indices,
        outside_mask=outside_mask_local,
    )


def _create_cache(
    src_mesh: dmesh.Mesh,
    resolution: tuple[int, ...],
    bounds: tuple[tuple[float, float], ...],
) -> InterpolationCache:
    """Build cached point ownership data for serial or MPI runs."""
    if src_mesh.comm.size == 1:
        return _create_serial_cache(src_mesh, resolution, bounds)
    return _create_parallel_cache(src_mesh, resolution, bounds)


def _evaluate_points(
    func: fem.Function, points: np.ndarray, cells: np.ndarray
) -> np.ndarray:
    """Evaluate a function on points owned by the current rank."""
    value_size = func.function_space.value_size
    if points.shape[0] == 0:
        return np.empty((0, value_size), dtype=func.dtype)

    valid = cells >= 0
    if valid.all():
        values = np.asarray(func.eval(points, cells))
        return values.reshape(points.shape[0], value_size)

    result = np.full((points.shape[0], value_size), np.nan, dtype=np.float64)
    if valid.any():
        valid_values = np.asarray(func.eval(points[valid], cells[valid]))
        result[valid] = valid_values.reshape(int(valid.sum()), value_size)
    return result


def _gather_parallel_values(
    comm: MPI.Intracomm, cache: InterpolationCache, local_values: np.ndarray
) -> np.ndarray:
    """Gather MPI-owned point evaluations back to rank 0 and broadcast the grid."""
    gathered = comm.gather(local_values, root=0)
    root_values: np.ndarray | None = None

    if comm.rank == 0:
        if cache.root_point_indices is None:
            raise RuntimeError(
                "Parallel interpolation cache missing root point indices."
            )
        value_size = local_values.shape[1]
        root_values = np.full(
            (cache.total_points, value_size), np.nan, dtype=local_values.dtype
        )
        for owner, owner_values in enumerate(gathered):
            point_indices = cache.root_point_indices[owner]
            if owner_values.shape[0] != point_indices.size:
                raise RuntimeError(
                    "Mismatch between point ownership and gathered interpolation values."
                )
            root_values[point_indices] = owner_values

    return comm.bcast(root_values, root=0)


def function_to_array(
    func: fem.Function,
    resolution: tuple[int, ...],
    bounds: tuple[tuple[float, float], ...] | None = None,
    cache: InterpolationCache | None = None,
) -> tuple[np.ndarray, InterpolationCache]:
    """Interpolate a scalar DOLFINx function onto a regular grid.

    Args:
        func: A scalar DOLFINx Function to evaluate.
        resolution: Grid resolution, e.g. (nx, ny) for 2D or (nx, ny, nz) for 3D.
        bounds: Domain bounds. If None, inferred from the mesh geometry.
        cache: Reusable interpolation cache. If None, one is created.

    Returns:
        Tuple of (array of shape `resolution`, cache for reuse).
    """
    result, cache = function_to_grid(func, resolution, bounds=bounds, cache=cache)
    if result.ndim != len(cache.resolution):
        raise ValueError(
            f"function_to_array expected a scalar function but got grid shape "
            f"{result.shape}."
        )
    return result, cache


def function_to_grid(
    func: fem.Function,
    resolution: tuple[int, ...],
    bounds: tuple[tuple[float, float], ...] | None = None,
    cache: InterpolationCache | None = None,
) -> tuple[np.ndarray, InterpolationCache]:
    """Interpolate a scalar or vector DOLFINx function onto a regular grid.

    Returns a scalar grid with shape `resolution`, or a vector grid with shape
    `(num_components, *resolution)`.
    """
    src_mesh = func.function_space.mesh

    if bounds is None:
        bounds = _global_bounds(src_mesh)

    if cache is None:
        cache = _create_cache(src_mesh, resolution, bounds)

    values = _evaluate_points(func, cache.points, cache.cells)
    if src_mesh.comm.size > 1:
        values = _gather_parallel_values(src_mesh.comm, cache, values)
    result = _reshape_grid_values(values, cache.resolution)

    return result, cache


def _reshape_grid_values(values: np.ndarray, resolution: tuple[int, ...]) -> np.ndarray:
    """Reshape `Function.eval()` output into scalar or vector grid arrays."""
    if values.ndim == 1:
        return values.reshape(resolution)

    value_shape = values.shape[1:]
    if value_shape == (1,):
        return values[:, 0].reshape(resolution)

    grid = values.reshape(*resolution, *value_shape)
    component_axes = tuple(range(len(resolution), grid.ndim))
    return np.moveaxis(grid, component_axes, tuple(range(len(component_axes))))

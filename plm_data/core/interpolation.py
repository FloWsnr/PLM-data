"""Interpolate DOLFINx functions onto regular grids for data output.

Uses direct point evaluation via Function.eval() with a precomputed
bounding-box-tree cell lookup. The grid points and their owning cells
are computed once and reused for every subsequent frame.
"""

from dataclasses import dataclass

import numpy as np
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.geometry import bb_tree, compute_colliding_cells, compute_collisions_points


@dataclass
class InterpolationCache:
    """Cached grid points and cell assignments for direct point evaluation.

    Created once per (source mesh, output resolution) combination.
    Reused across all frames; only `func.eval()` is called per frame.
    """

    points: np.ndarray  # (N, 3) evaluation points
    cells: np.ndarray  # (N,) owning cell index per point
    resolution: tuple[int, ...]
    gdim: int


def _create_cache(
    src_mesh: dmesh.Mesh,
    resolution: tuple[int, ...],
    bounds: tuple[tuple[float, float], ...],
) -> InterpolationCache:
    """Build the regular grid and find owning cells via bounding-box tree."""
    gdim = src_mesh.geometry.dim

    # Build regular grid points
    axes = [np.linspace(bounds[d][0], bounds[d][1], resolution[d]) for d in range(gdim)]
    grids = np.meshgrid(*axes, indexing="ij")
    flat = [g.ravel() for g in grids]

    # Function.eval() always expects (N, 3)
    n_pts = flat[0].size
    points = np.zeros((n_pts, 3), dtype=src_mesh.geometry.x.dtype)
    for d in range(gdim):
        points[:, d] = flat[d]

    # Find owning cells
    tree = bb_tree(src_mesh, src_mesh.topology.dim)
    candidates = compute_collisions_points(tree, points)
    colliding = compute_colliding_cells(src_mesh, candidates, points)

    cells = np.empty(n_pts, dtype=np.int32)
    for i in range(n_pts):
        links = colliding.links(i)
        cells[i] = links[0] if links.size > 0 else -1

    missing = int((cells < 0).sum())
    if missing > 0:
        raise RuntimeError(
            f"{missing}/{n_pts} grid points found no owning cell. "
            "Check domain bounds vs mesh geometry."
        )

    return InterpolationCache(
        points=points,
        cells=cells,
        resolution=resolution,
        gdim=gdim,
    )


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
    gdim = src_mesh.geometry.dim

    if bounds is None:
        coords = src_mesh.geometry.x
        bounds = tuple(
            (float(coords[:, d].min()), float(coords[:, d].max())) for d in range(gdim)
        )

    if cache is None:
        cache = _create_cache(src_mesh, resolution, bounds)

    values = func.eval(cache.points, cache.cells)
    result = _reshape_grid_values(values, cache.resolution)

    nan_count = int(np.isnan(result).sum())
    if nan_count > 0:
        raise RuntimeError(
            f"Interpolation produced {nan_count}/{result.size} NaN values. "
            "This may indicate solver divergence or points outside the mesh."
        )
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

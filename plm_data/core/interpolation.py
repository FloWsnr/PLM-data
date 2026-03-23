"""Interpolate DOLFINx functions onto regular grids for data output."""

import numpy as np
from dolfinx import fem, geometry


def function_to_array(
    func: fem.Function,
    resolution: tuple[int, ...],
    bounds: tuple[tuple[float, float], ...] | None = None,
) -> np.ndarray:
    """Interpolate a DOLFINx function onto a regular grid.

    Args:
        func: A DOLFINx Function to evaluate.
        resolution: Grid resolution, e.g. (nx, ny) for 2D.
        bounds: Domain bounds as ((x_min, x_max), (y_min, y_max), ...).
            If None, inferred from the mesh geometry.

    Returns:
        Array of shape `resolution` with function values on the regular grid.
    """
    msh = func.function_space.mesh
    gdim = msh.geometry.dim

    # Infer bounds from mesh if not provided
    if bounds is None:
        coords = msh.geometry.x
        bounds = tuple(
            (float(coords[:, d].min()), float(coords[:, d].max()))
            for d in range(gdim)
        )

    # Create regular grid points
    axes = [
        np.linspace(bounds[d][0], bounds[d][1], resolution[d])
        for d in range(gdim)
    ]

    if gdim == 2:
        # axes[0] -> x (columns), axes[1] -> y (rows)
        xx, yy = np.meshgrid(axes[0], axes[1])
        points = np.zeros((xx.size, 3))
        points[:, 0] = xx.ravel()
        points[:, 1] = yy.ravel()
    elif gdim == 3:
        xx, yy, zz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        points = np.zeros((xx.size, 3))
        points[:, 0] = xx.ravel()
        points[:, 1] = yy.ravel()
        points[:, 2] = zz.ravel()
    else:
        raise ValueError(f"Unsupported geometry dimension: {gdim}")

    # Find cells containing each point
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    cell_collisions = geometry.compute_colliding_cells(msh, cell_candidates, points)

    # Evaluate function point by point
    values = np.full(points.shape[0], np.nan)
    for i in range(points.shape[0]):
        cells = cell_collisions.links(i)
        if len(cells) > 0:
            val = func.eval(points[i : i + 1], cells[:1])
            values[i] = val.flat[0]

    return values.reshape(resolution)

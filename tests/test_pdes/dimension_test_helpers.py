"""Helper utilities for testing PDE dimension support."""

import numpy as np
from pde import CartesianGrid, FieldCollection, ScalarField

from pde_sim.core.config import BoundaryConfig


def create_grid_for_dimension(ndim: int, resolution: int = 16, periodic: bool = True) -> CartesianGrid:
    """Create a grid for the given number of dimensions.

    Args:
        ndim: Number of dimensions (1, 2, or 3).
        resolution: Grid resolution per dimension.
        periodic: Whether to use periodic boundaries.

    Returns:
        CartesianGrid of the appropriate dimension.
    """
    if ndim == 1:
        return CartesianGrid([[0, 1]], [resolution], periodic=[periodic])
    elif ndim == 2:
        return CartesianGrid([[0, 1], [0, 1]], [resolution, resolution], periodic=[periodic, periodic])
    elif ndim == 3:
        return CartesianGrid([[0, 1], [0, 1], [0, 1]], [resolution, resolution, resolution], periodic=[periodic, periodic, periodic])
    else:
        raise ValueError(f"Unsupported dimension: {ndim}")


def create_bc_for_dimension(ndim: int, periodic: bool = True) -> BoundaryConfig:
    """Create boundary conditions for the given number of dimensions.

    Args:
        ndim: Number of dimensions (1, 2, or 3).
        periodic: Whether to use periodic boundaries.

    Returns:
        BoundaryConfig for the appropriate dimension.
    """
    bc_type = "periodic" if periodic else "neumann:0"

    if ndim == 1:
        return BoundaryConfig(x_minus=bc_type, x_plus=bc_type)
    elif ndim == 2:
        return BoundaryConfig(x_minus=bc_type, x_plus=bc_type, y_minus=bc_type, y_plus=bc_type)
    elif ndim == 3:
        return BoundaryConfig(x_minus=bc_type, x_plus=bc_type, y_minus=bc_type, y_plus=bc_type, z_minus=bc_type, z_plus=bc_type)
    else:
        raise ValueError(f"Unsupported dimension: {ndim}")


def run_dimension_test(
    preset,
    ndim: int,
    t_end: float = 0.001,
    dt: float = 0.0001,
    resolution: int | None = None,
    ic_type: str = "random-uniform",
    ic_params: dict | None = None,
    periodic: bool = True,
    seed: int = 42,
) -> tuple[ScalarField | FieldCollection, bool]:
    """Run a short simulation to test dimension support.

    Args:
        preset: The PDE preset instance.
        ndim: Number of dimensions to test.
        t_end: Simulation end time.
        dt: Time step.
        resolution: Grid resolution (defaults to 32 for 1D, 16 for 2D, 8 for 3D).
        ic_type: Initial condition type.
        ic_params: Initial condition parameters.
        periodic: Whether to use periodic boundaries.
        seed: Random seed.

    Returns:
        Tuple of (result, is_finite) where result is the final state
        and is_finite indicates if all values are finite.
    """
    np.random.seed(seed)

    # Set resolution based on dimension if not specified
    if resolution is None:
        resolution = {1: 32, 2: 16, 3: 8}[ndim]

    # Default IC params
    if ic_params is None:
        ic_params = {"low": 0.1, "high": 0.9}

    grid = create_grid_for_dimension(ndim, resolution=resolution, periodic=periodic)
    bc = create_bc_for_dimension(ndim, periodic=periodic)

    params = preset.get_default_parameters()
    pde = preset.create_pde(params, bc, grid)

    state = preset.create_initial_state(grid=grid, ic_type=ic_type, ic_params=ic_params)

    result = pde.solve(state, t_range=t_end, dt=dt, solver="euler", tracker=None)

    # Check if result is finite
    if isinstance(result, ScalarField):
        is_finite = np.isfinite(result.data).all()
    else:
        is_finite = all(np.isfinite(field.data).all() for field in result)

    return result, is_finite


def check_result_finite(result: ScalarField | FieldCollection, preset_name: str, ndim: int) -> None:
    """Assert that simulation result contains only finite values.

    Args:
        result: The simulation result.
        preset_name: Name of the PDE preset (for error message).
        ndim: Number of dimensions (for error message).

    Raises:
        AssertionError: If result contains non-finite values.
    """
    if isinstance(result, ScalarField):
        assert np.isfinite(result.data).all(), f"{preset_name} in {ndim}D produced non-finite values"
    else:
        for i, field in enumerate(result):
            assert np.isfinite(field.data).all(), f"{preset_name} in {ndim}D field {i} produced non-finite values"

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
    params: dict,
    num_steps: int = 5,
    dt: float = 0.001,
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
        params: Dictionary of PDE parameters.
        num_steps: Number of time steps to run (default 5 for speed).
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

    pde = preset.create_pde(params, bc, grid)

    state = preset.create_initial_state(grid=grid, ic_type=ic_type, ic_params=ic_params)

    # Compute t_end based on number of steps
    t_end = num_steps * dt

    result = pde.solve(state, t_range=t_end, dt=dt, solver="euler", tracker=None, backend="numpy")

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


def _compute_axis_variation(data: np.ndarray, axis: int) -> float:
    """Compute mean standard deviation along an axis.

    Args:
        data: The data array to analyze.
        axis: The axis along which to compute standard deviation.

    Returns:
        Mean standard deviation across slices perpendicular to the axis.
    """
    return float(np.mean(np.std(data, axis=axis)))


def _check_field_dimension_variation(
    data: np.ndarray,
    field_name: str,
    expected_ndim: int,
    preset_name: str,
    relative_tol: float = 1e-6,
    absolute_tol: float = 1e-14,
) -> None:
    """Check that a single field has variation in all expected dimensions.

    Args:
        data: The field data array.
        field_name: Name of the field (for error message).
        expected_ndim: Expected number of spatial dimensions.
        preset_name: Name of the PDE preset (for error message).
        relative_tol: Relative tolerance for variation threshold.
        absolute_tol: Absolute tolerance floor for variation threshold.

    Raises:
        AssertionError: If the field lacks variation in any dimension.
    """
    data_range = data.max() - data.min()
    threshold = max(data_range * relative_tol, absolute_tol)

    axis_names = ["x", "y", "z"][:expected_ndim]

    for axis in range(expected_ndim):
        variation = _compute_axis_variation(data, axis)
        if variation < threshold:
            raise AssertionError(
                f"{preset_name} {field_name} in {expected_ndim}D: "
                f"No variation along {axis_names[axis]}-axis (std={variation:.2e}). "
                f"This suggests broadcasting of lower-dimensional data."
            )


def check_dimension_variation(
    result: ScalarField | FieldCollection,
    expected_ndim: int,
    preset_name: str,
    relative_tol: float = 1e-6,
    absolute_tol: float = 1e-14,
) -> None:
    """Assert simulation result varies in all expected dimensions.

    Catches PDEs that claim multi-D support but broadcast lower-D data.

    Args:
        result: The simulation result.
        expected_ndim: Expected number of spatial dimensions.
        preset_name: Name of the PDE preset (for error message).
        relative_tol: Relative tolerance for variation threshold.
        absolute_tol: Absolute tolerance floor for variation threshold.

    Raises:
        AssertionError: If result lacks variation in any dimension.
    """
    if isinstance(result, ScalarField):
        _check_field_dimension_variation(
            result.data,
            result.label or "field",
            expected_ndim,
            preset_name,
            relative_tol,
            absolute_tol,
        )
    else:
        for i, field in enumerate(result):
            _check_field_dimension_variation(
                field.data,
                field.label or f"field_{i}",
                expected_ndim,
                preset_name,
                relative_tol,
                absolute_tol,
            )


def check_result_dimensions(
    result: ScalarField | FieldCollection,
    expected_ndim: int,
    expected_resolution: int | list[int],
    preset_name: str,
) -> None:
    """Assert that simulation result has the correct dimensions and shape.

    This verifies that:
    1. The grid has the expected number of dimensions
    2. The data array has the expected shape
    3. Each field (for multi-field PDEs) has consistent dimensions

    Args:
        result: The simulation result.
        expected_ndim: Expected number of spatial dimensions (1, 2, or 3).
        expected_resolution: Expected resolution per dimension.
        preset_name: Name of the PDE preset (for error message).

    Raises:
        AssertionError: If dimensions don't match expected values.
    """
    # Normalize resolution to list
    if isinstance(expected_resolution, int):
        expected_shape = tuple([expected_resolution] * expected_ndim)
    else:
        expected_shape = tuple(expected_resolution)

    if isinstance(result, ScalarField):
        # Check grid dimensionality
        assert result.grid.dim == expected_ndim, (
            f"{preset_name}: grid.dim={result.grid.dim}, expected {expected_ndim}"
        )
        # Check data shape matches expected dimensions
        assert result.data.ndim == expected_ndim, (
            f"{preset_name}: data.ndim={result.data.ndim}, expected {expected_ndim}"
        )
        assert result.data.shape == expected_shape, (
            f"{preset_name}: data.shape={result.data.shape}, expected {expected_shape}"
        )
    else:
        # FieldCollection - check each field
        for i, field in enumerate(result):
            assert field.grid.dim == expected_ndim, (
                f"{preset_name} field {i}: grid.dim={field.grid.dim}, expected {expected_ndim}"
            )
            assert field.data.ndim == expected_ndim, (
                f"{preset_name} field {i}: data.ndim={field.data.ndim}, expected {expected_ndim}"
            )
            assert field.data.shape == expected_shape, (
                f"{preset_name} field {i}: data.shape={field.data.shape}, expected {expected_shape}"
            )

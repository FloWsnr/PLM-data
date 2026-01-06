"""Tests for different compute backends."""

import numpy as np
import pytest

from pde_sim.pdes import get_pde_preset


class TestBackends:
    """Tests for different compute backends."""

    def test_numba_backend(self, small_grid):
        """Test running simulation with numba backend."""
        preset = get_pde_preset("heat")
        params = {"D_T": 0.01}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "gaussian-blobs", {"num_blobs": 1, "amplitude": 1.0}
        )

        # Test with numba backend
        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", backend="numba")

        assert np.all(np.isfinite(result.data))

    def test_numpy_backend(self, small_grid):
        """Test running simulation with numpy backend."""
        preset = get_pde_preset("heat")
        params = {"D_T": 0.01}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid, "gaussian-blobs", {"num_blobs": 1, "amplitude": 1.0}
        )

        # Test with numpy backend
        result = pde.solve(state, t_range=0.001, dt=0.0001, solver="euler", backend="numpy")

        assert np.all(np.isfinite(result.data))

"""Tests for Compressible Navier-Stokes PDE."""

import numpy as np
import pytest

from pde import FieldCollection

from pde_sim.pdes import get_pde_preset
from tests.test_pdes.dimension_test_helpers import (
    create_grid_for_dimension,
    create_bc_for_dimension,
    check_result_finite,
    check_dimension_variation,
)


class TestCompressibleNavierStokesPDE:
    """Tests for Compressible Navier-Stokes PDE."""

    def test_metadata(self):
        """Test that metadata is correctly defined."""
        preset = get_pde_preset("compressible-navier-stokes")
        meta = preset.metadata

        assert meta.name == "compressible-navier-stokes"
        assert meta.category == "fluids"
        assert meta.num_fields == 4
        assert meta.field_names == ["rho", "u", "v", "p"]
        assert meta.supported_dimensions == [2]

    def test_short_simulation(self):
        """Test running a short simulation with acoustic-pulse IC."""
        preset = get_pde_preset("compressible-navier-stokes")
        preset.validate_dimension(2)

        grid = create_grid_for_dimension(2, resolution=16)
        bc = create_bc_for_dimension(2)

        params = {"gamma": 1.4, "mu": 0.01, "kappa": 0.01}
        pde = preset.create_pde(params, bc, grid)

        state = preset.create_initial_state(
            grid, "acoustic-pulse",
            {"x0": 0.5, "y0": 0.5, "amplitude": 0.1, "width": 0.05, "seed": 42},
        )

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "compressible-navier-stokes", 2)
        check_dimension_variation(result, 2, "compressible-navier-stokes")

    def test_kelvin_helmholtz_ic(self):
        """Test running with Kelvin-Helmholtz IC."""
        preset = get_pde_preset("compressible-navier-stokes")

        grid = create_grid_for_dimension(2, resolution=16)
        bc = create_bc_for_dimension(2)

        params = {"gamma": 1.4, "mu": 0.01, "kappa": 0.01}
        pde = preset.create_pde(params, bc, grid)

        state = preset.create_initial_state(
            grid, "kelvin-helmholtz",
            {"shear_y": 0.5, "shear_width": 0.05, "velocity_amplitude": 0.3, "seed": 42},
        )

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "compressible-navier-stokes", 2)

    def test_density_blob_ic(self):
        """Test running with density-blob IC."""
        preset = get_pde_preset("compressible-navier-stokes")

        grid = create_grid_for_dimension(2, resolution=16)
        bc = create_bc_for_dimension(2)

        params = {"gamma": 1.4, "mu": 0.01, "kappa": 0.01}
        pde = preset.create_pde(params, bc, grid)

        state = preset.create_initial_state(
            grid, "density-blob",
            {"x0": 0.5, "y0": 0.5, "amplitude": 0.3, "width": 0.08, "seed": 42},
        )

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "compressible-navier-stokes", 2)
        check_dimension_variation(result, 2, "compressible-navier-stokes")

    def test_shock_tube_ic(self):
        """Test running with shock-tube IC (Sod problem)."""
        preset = get_pde_preset("compressible-navier-stokes")

        grid = create_grid_for_dimension(2, resolution=16)
        bc = create_bc_for_dimension(2)

        params = {"gamma": 1.4, "mu": 0.01, "kappa": 0.01}
        pde = preset.create_pde(params, bc, grid)

        state = preset.create_initial_state(
            grid, "shock-tube",
            {
                "orientation": "vertical",
                "interface_pos": 0.5,
                "rho_left": 1.0,
                "rho_right": 0.125,
                "p_left": 1.0,
                "p_right": 0.1,
                "smooth": 0.02,
                "seed": 42,
            },
        )

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "compressible-navier-stokes", 2)
        # Shock tube is intentionally 1D-like (uniform along y), so skip y-variation check

    def test_colliding_jets_ic(self):
        """Test running with colliding-jets IC."""
        preset = get_pde_preset("compressible-navier-stokes")

        grid = create_grid_for_dimension(2, resolution=16)
        bc = create_bc_for_dimension(2)

        params = {"gamma": 1.4, "mu": 0.01, "kappa": 0.01}
        pde = preset.create_pde(params, bc, grid)

        state = preset.create_initial_state(
            grid, "colliding-jets",
            {
                "jet_y": 0.5,
                "jet_width": 0.05,
                "jet_velocity": 0.5,
                "rho_0": 1.0,
                "p_0": 1.0,
                "seed": 42,
            },
        )

        result = pde.solve(state, t_range=0.01, dt=0.001, solver="euler", tracker=None, backend="numpy")

        assert isinstance(result, FieldCollection)
        check_result_finite(result, "compressible-navier-stokes", 2)
        check_dimension_variation(result, 2, "compressible-navier-stokes")

    def test_unsupported_dimensions(self):
        """Test that compressible-navier-stokes only supports 2D."""
        preset = get_pde_preset("compressible-navier-stokes")

        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(1)
        with pytest.raises(ValueError, match="does not support"):
            preset.validate_dimension(3)

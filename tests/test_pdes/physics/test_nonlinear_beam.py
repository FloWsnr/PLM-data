"""Tests for Nonlinear Beam PDE."""

import numpy as np
import pytest

from pde import CartesianGrid, FieldCollection

from pde_sim.pdes import get_pde_preset, list_presets
from pde_sim.pdes.physics.nonlinear_beam import NonlinearBeamEquationPDE


@pytest.fixture
def small_grid():
    """Create a small grid for fast tests."""
    return CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=True)


@pytest.fixture
def non_periodic_grid():
    """Create a small grid with non-periodic BCs for testing."""
    return CartesianGrid([[0, 10], [0, 10]], [16, 16], periodic=False)


class TestNonlinearBeamPDE:
    """Tests for Nonlinear Beam PDE."""

    def test_registered(self):
        """Test that nonlinear-beam is registered."""
        assert "nonlinear-beam" in list_presets()

    def test_metadata(self):
        """Test metadata."""
        preset = get_pde_preset("nonlinear-beam")
        meta = preset.metadata

        assert meta.name == "nonlinear-beam"
        assert meta.category == "physics"
        assert meta.num_fields == 3
        assert meta.field_names == ["u", "v", "w"]

    def test_create_pde(self, small_grid):
        """Test creating the PDE object."""
        preset = get_pde_preset("nonlinear-beam")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)

        assert pde is not None
        assert isinstance(pde, NonlinearBeamEquationPDE)
        assert pde.E_star == params["E_star"]
        assert pde.Delta_E == params["Delta_E"]
        assert pde.eps == params["eps"]

    def test_create_initial_state(self, small_grid):
        """Test creating initial state returns FieldCollection with 3 fields."""
        preset = get_pde_preset("nonlinear-beam")
        params = preset.get_default_parameters()
        bc = {"x": "periodic", "y": "periodic"}

        state = preset.create_initial_state(
            small_grid, "default", {"seed": 42}, parameters=params, bc=bc
        )

        assert isinstance(state, FieldCollection)
        assert len(state) == 3
        assert state[0].label == "u"
        assert state[1].label == "v"
        assert state[2].label == "w"
        assert all(np.isfinite(f.data).all() for f in state)

    def test_initial_state_consistency(self, small_grid):
        """Test that v and w are computed correctly from u."""
        preset = get_pde_preset("nonlinear-beam")
        params = {"E_star": 0.0001, "Delta_E": 10.0, "eps": 0.1}
        bc = {"x": "periodic", "y": "periodic"}

        state = preset.create_initial_state(
            small_grid, "default", {"seed": 42}, parameters=params, bc=bc
        )

        u, v, w = state

        # v should equal laplace(u)
        bc_spec = preset._convert_bc(bc)
        expected_v = u.laplace(bc=bc_spec).data
        np.testing.assert_allclose(v.data, expected_v, rtol=1e-10)

        # w should equal E(v) * v
        E_v = params["E_star"] + params["Delta_E"] * (
            1.0 + np.tanh(v.data / params["eps"])
        ) / 2.0
        expected_w = E_v * v.data
        np.testing.assert_allclose(w.data, expected_w, rtol=1e-10)

    def test_short_simulation(self, small_grid):
        """Test running a short simulation."""
        preset = get_pde_preset("nonlinear-beam")
        # Use smaller stiffness parameters for stability
        params = {"E_star": 0.0001, "Delta_E": 1.0, "eps": 0.1}
        # Use periodic BC to match the periodic grid
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid,
            "default",
            {"amplitude": 0.1, "seed": 42},
            parameters=params,
            bc=bc,
        )

        # Run a very short simulation with tiny timestep (fourth-order PDE is stiff)
        result = pde.solve(state, t_range=0.0001, dt=1e-7)

        # Check that result is finite and valid (result is FieldCollection)
        assert result is not None
        assert isinstance(result, FieldCollection)
        assert len(result) == 3
        assert all(
            np.isfinite(f.data).all() for f in result
        ), "Simulation produced NaN or Inf values"

    def test_evolution_rate(self, small_grid):
        """Test that evolution_rate computes the correct structure."""
        preset = get_pde_preset("nonlinear-beam")
        params = {"E_star": 0.0001, "Delta_E": 10.0, "eps": 0.1}
        bc = {"x": "periodic", "y": "periodic"}

        pde = preset.create_pde(params, bc, small_grid)
        state = preset.create_initial_state(
            small_grid,
            "default",
            {"amplitude": 0.5, "seed": 42},
            parameters=params,
            bc=bc,
        )

        # Compute evolution rate
        rhs = pde.evolution_rate(state, t=0)

        # Check structure
        assert isinstance(rhs, FieldCollection)
        assert len(rhs) == 3

        # du/dt should be non-zero (indicates dynamics)
        assert np.any(rhs[0].data != 0)

        # dv/dt and dw/dt should be zero (algebraic constraints)
        np.testing.assert_allclose(rhs[1].data, 0)
        np.testing.assert_allclose(rhs[2].data, 0)

        # All fields should be finite
        assert all(np.isfinite(f.data).all() for f in rhs)

    def test_get_equations_for_metadata(self):
        """Test that equations are properly formatted."""
        preset = get_pde_preset("nonlinear-beam")
        params = {"E_star": 0.001, "Delta_E": 5.0, "eps": 0.05}

        equations = preset.get_equations_for_metadata(params)

        assert "u" in equations
        assert "v" in equations
        assert "w" in equations
        assert equations["u"] == "-laplace(w)"
        assert equations["v"] == "laplace(u)"
        # w equation should include parameter values
        assert "0.001" in equations["w"]
        assert "5.0" in equations["w"]
        assert "0.05" in equations["w"]

"""Tests for running actual PDE preset simulations."""

import numpy as np

from plm_data.core.config import (
    BCConfig,
    DomainConfig,
    ICConfig,
    OutputConfig,
    SimulationConfig,
    SolverConfig,
    SourceTermConfig,
)
from plm_data.core.output import FrameWriter
from plm_data.presets import get_preset


def test_heat_preset_single_step(tmp_path, heat_config):
    preset = get_preset("heat")
    output_dir = tmp_path / "basic" / "heat"
    writer = FrameWriter(output_dir, heat_config)

    result = preset.run(heat_config, writer)

    assert result.solver_converged is True
    assert result.num_timesteps == 1
    assert result.num_dofs > 0

    # Initial frame + 1 timestep frame = 2 frames
    assert writer.frame_count == 2
    writer.finalize()
    npy_path = output_dir / "u.npy"
    assert npy_path.exists()
    arr = np.load(npy_path)
    assert arr.shape == (2, *heat_config.output_resolution)


def test_poisson_preset(tmp_path, poisson_config):
    preset = get_preset("poisson")
    output_dir = tmp_path / "basic" / "poisson"
    writer = FrameWriter(output_dir, poisson_config)

    result = preset.run(poisson_config, writer)

    assert result.solver_converged is True
    assert result.num_dofs > 0
    assert writer.frame_count == 1
    writer.finalize()
    npy_path = output_dir / "u.npy"
    assert npy_path.exists()
    arr = np.load(npy_path)
    assert arr.shape == (1, *poisson_config.output_resolution)
    # Poisson with sin source on unit square should have positive interior values
    assert np.max(arr) > 0


def test_cahn_hilliard_preset_single_step(tmp_path, cahn_hilliard_config):
    preset = get_preset("cahn_hilliard")
    output_dir = tmp_path / "physics" / "cahn_hilliard"
    writer = FrameWriter(output_dir, cahn_hilliard_config)

    result = preset.run(cahn_hilliard_config, writer)

    assert result.solver_converged is True
    assert result.num_timesteps == 1
    assert result.num_dofs > 0

    # Initial frame + 1 timestep frame = 2 frames
    assert writer.frame_count == 2
    writer.finalize()
    npy_path = output_dir / "c.npy"
    assert npy_path.exists()
    arr = np.load(npy_path)
    assert arr.shape == (2, *cahn_hilliard_config.output_resolution)
    # Concentration should be around the initial mean (0.63)
    assert 0.0 < np.mean(arr) < 1.0


def test_navier_stokes_preset_single_step(tmp_path, navier_stokes_config):
    preset = get_preset("navier_stokes")
    output_dir = tmp_path / "fluids" / "navier_stokes"
    writer = FrameWriter(output_dir, navier_stokes_config)

    result = preset.run(navier_stokes_config, writer)

    assert result.solver_converged is True
    assert result.num_timesteps == 1
    assert result.num_dofs > 0

    # Initial frame + 1 timestep frame = 2 frames
    assert writer.frame_count == 2
    writer.finalize()

    # Check all three output fields exist with correct shapes
    for field in ["velocity_x", "velocity_y", "pressure"]:
        npy_path = output_dir / f"{field}.npy"
        assert npy_path.exists()
        arr = np.load(npy_path)
        assert arr.shape == (2, *navier_stokes_config.output_resolution)

    # Lid-driven cavity: velocity_x should have non-zero values from the lid
    vx = np.load(output_dir / "velocity_x.npy")
    assert np.max(np.abs(vx)) > 0


def _make_ns_config(
    tmp_path, *, initial_conditions, source_terms=None, parameters=None
):
    """Helper to build a minimal NS config with custom ICs/sources."""
    domain = DomainConfig(
        type="rectangle",
        params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
    )
    if parameters is None:
        parameters = {"Re": 25.0, "k": 1.0}
    if source_terms is None:
        source_terms = {
            "velocity": SourceTermConfig(type="none", params={}),
            "pressure": SourceTermConfig(type="none", params={}),
        }
    return SimulationConfig(
        preset="navier_stokes",
        parameters=parameters,
        domain=domain,
        output_resolution=[4, 4],
        boundary_conditions={
            "velocity": {
                "x-": BCConfig(type="dirichlet", value=[0.0, 0.0]),
                "x+": BCConfig(type="dirichlet", value=[0.0, 0.0]),
                "y-": BCConfig(type="dirichlet", value=[0.0, 0.0]),
                "y+": BCConfig(type="dirichlet", value=[1.0, 0.0]),
            },
            "pressure": {},
        },
        source_terms=source_terms,
        initial_conditions=initial_conditions,
        output=OutputConfig(path=tmp_path, num_frames=2, formats=["numpy"]),
        solver=SolverConfig(
            options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_14": "80",
                "mat_mumps_icntl_24": "1",
                "mat_mumps_icntl_25": "0",
                "ksp_error_if_not_converged": "1",
            }
        ),
        dt=0.1,
        t_end=0.1,
        seed=42,
    )


def _run_ns_preset(tmp_path, config):
    """Run NS preset and return (result, output_dir)."""
    preset = get_preset("navier_stokes")
    output_dir = tmp_path / "fluids" / "navier_stokes"
    writer = FrameWriter(output_dir, config)
    result = preset.run(config, writer)
    writer.finalize()
    return result, output_dir


def test_navier_stokes_per_component_ic(tmp_path):
    """Test NS with per-component constant ICs (skips Stokes solve)."""
    config = _make_ns_config(
        tmp_path,
        initial_conditions={
            "velocity_x": ICConfig(type="constant", params={"value": 0.0}),
            "velocity_y": ICConfig(type="constant", params={"value": 0.0}),
        },
    )
    result, output_dir = _run_ns_preset(tmp_path, config)

    assert result.solver_converged is True
    assert result.num_timesteps == 1
    for field in ["velocity_x", "velocity_y", "pressure"]:
        arr = np.load(output_dir / f"{field}.npy")
        assert arr.shape == (2, *config.output_resolution)


def test_navier_stokes_gaussian_bump_ic(tmp_path):
    """Test NS with gaussian_bump IC on velocity_x component."""
    config = _make_ns_config(
        tmp_path,
        initial_conditions={
            "velocity_x": ICConfig(
                type="gaussian_bump",
                params={"sigma": 0.2, "amplitude": 0.5, "cx": 0.5, "cy": 0.5},
            ),
            "velocity_y": ICConfig(type="constant", params={"value": 0.0}),
        },
    )
    result, output_dir = _run_ns_preset(tmp_path, config)

    assert result.solver_converged is True
    vx = np.load(output_dir / "velocity_x.npy")
    assert vx.shape == (2, *config.output_resolution)
    # Initial frame should have non-zero vx from the gaussian bump
    assert np.max(np.abs(vx[0])) > 0


def test_navier_stokes_body_force(tmp_path):
    """Test NS with custom body force (gravity-like)."""
    config = _make_ns_config(
        tmp_path,
        parameters={"Re": 25.0, "k": 1.0, "force_x": 0.0, "force_y": -1.0},
        initial_conditions={
            "velocity": ICConfig(type="custom", params={}),
        },
        source_terms={
            "velocity": SourceTermConfig(type="custom", params={}),
            "pressure": SourceTermConfig(type="none", params={}),
        },
    )
    result, output_dir = _run_ns_preset(tmp_path, config)

    assert result.solver_converged is True
    # Body force in y-direction should produce non-zero velocity_y
    vy = np.load(output_dir / "velocity_y.npy")
    assert np.max(np.abs(vy)) > 0


def test_cahn_hilliard_constant_ic(tmp_path):
    """Test Cahn-Hilliard with constant IC instead of random_perturbation."""
    config = SimulationConfig(
        preset="cahn_hilliard",
        parameters={"lmbda": 0.01, "theta": 0.5},
        domain=DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
        ),
        output_resolution=[4, 4],
        boundary_conditions={"c": {}},
        source_terms={"c": SourceTermConfig(type="none", params={})},
        initial_conditions={
            "c": ICConfig(type="constant", params={"value": 0.5}),
        },
        output=OutputConfig(path=tmp_path, num_frames=2, formats=["numpy"]),
        solver=SolverConfig(
            options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "none",
                "ksp_type": "preonly",
                "pc_type": "lu",
            }
        ),
        dt=5e-6,
        t_end=5e-6,
        seed=42,
    )

    preset = get_preset("cahn_hilliard")
    output_dir = tmp_path / "physics" / "cahn_hilliard"
    writer = FrameWriter(output_dir, config)
    result = preset.run(config, writer)
    writer.finalize()

    assert result.solver_converged is True
    arr = np.load(output_dir / "c.npy")
    assert arr.shape == (2, *config.output_resolution)
    # With constant IC=0.5, concentration should stay near 0.5 (no phase separation)
    assert np.allclose(arr[0], 0.5, atol=0.05)

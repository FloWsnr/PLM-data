"""Tests for running actual PDE preset simulations."""

import numpy as np

from plm_data.core.config import (
    BoundaryConditionConfig,
    DomainConfig,
    FieldConfig,
    FieldExpressionConfig,
    FieldOutputConfig,
    OutputConfig,
    SimulationConfig,
    SolverConfig,
    TimeConfig,
)
from plm_data.core.output import FrameWriter
from plm_data.presets import get_preset


def constant(value):
    return FieldExpressionConfig(type="constant", params={"value": value})


def scalar_expr(expr_type: str, **params):
    return FieldExpressionConfig(type=expr_type, params=params)


def vector_expr(**components):
    return FieldExpressionConfig(components=components)


def _run_preset(config):
    preset = get_preset(config.preset)
    output_dir = config.output.path / preset.spec.category / preset.spec.name
    writer = FrameWriter(output_dir, config, preset.spec)
    result = preset.build_problem(config).run(writer)
    writer.finalize()
    return result, output_dir


def test_heat_preset_single_step(heat_config):
    result, output_dir = _run_preset(heat_config)
    assert result.solver_converged is True
    assert result.num_timesteps == 1
    assert result.num_dofs > 0

    arr = np.load(output_dir / "u.npy")
    assert arr.shape == (2, *heat_config.output.resolution)


def test_poisson_preset(poisson_config):
    result, output_dir = _run_preset(poisson_config)
    assert result.solver_converged is True
    assert result.num_dofs > 0

    arr = np.load(output_dir / "u.npy")
    assert arr.shape == (1, *poisson_config.output.resolution)
    assert np.max(arr) > 0


def test_cahn_hilliard_preset_single_step(cahn_hilliard_config):
    result, output_dir = _run_preset(cahn_hilliard_config)
    assert result.solver_converged is True
    assert result.num_timesteps == 1
    assert result.num_dofs > 0

    arr = np.load(output_dir / "c.npy")
    assert arr.shape == (2, *cahn_hilliard_config.output.resolution)
    assert 0.0 < np.mean(arr) < 1.0


def test_navier_stokes_preset_single_step(navier_stokes_config):
    result, output_dir = _run_preset(navier_stokes_config)
    assert result.solver_converged is True
    assert result.num_timesteps == 1
    assert result.num_dofs > 0

    for field in ["velocity_x", "velocity_y", "pressure"]:
        arr = np.load(output_dir / f"{field}.npy")
        assert arr.shape == (2, *navier_stokes_config.output.resolution)

    vx = np.load(output_dir / "velocity_x.npy")
    assert np.max(np.abs(vx)) > 0


def _make_ns_config(tmp_path, *, initial_condition, source=None, parameters=None):
    domain = DomainConfig(
        type="rectangle",
        params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
    )
    if parameters is None:
        parameters = {"Re": 25.0, "k": 1.0}
    if source is None:
        source = scalar_expr("none")

    return SimulationConfig(
        preset="navier_stokes",
        parameters=parameters,
        domain=domain,
        fields={
            "velocity": FieldConfig(
                boundary_conditions={
                    "x-": BoundaryConditionConfig(
                        type="dirichlet",
                        value=vector_expr(x=constant(0.0), y=constant(0.0)),
                    ),
                    "x+": BoundaryConditionConfig(
                        type="dirichlet",
                        value=vector_expr(x=constant(0.0), y=constant(0.0)),
                    ),
                    "y-": BoundaryConditionConfig(
                        type="dirichlet",
                        value=vector_expr(x=constant(0.0), y=constant(0.0)),
                    ),
                    "y+": BoundaryConditionConfig(
                        type="dirichlet",
                        value=vector_expr(x=constant(1.0), y=constant(0.0)),
                    ),
                },
                source=source,
                initial_condition=initial_condition,
                output=FieldOutputConfig(mode="components"),
            ),
            "pressure": FieldConfig(output=FieldOutputConfig(mode="scalar")),
        },
        output=OutputConfig(
            path=tmp_path, resolution=[4, 4], num_frames=2, formats=["numpy"]
        ),
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
        time=TimeConfig(dt=0.1, t_end=0.1),
        seed=42,
    )


def test_navier_stokes_per_component_ic(tmp_path):
    config = _make_ns_config(
        tmp_path,
        initial_condition=vector_expr(x=constant(0.0), y=constant(0.0)),
    )
    result, output_dir = _run_preset(config)
    assert result.solver_converged is True

    for field in ["velocity_x", "velocity_y", "pressure"]:
        arr = np.load(output_dir / f"{field}.npy")
        assert arr.shape == (2, *config.output.resolution)


def test_navier_stokes_gaussian_bump_ic(tmp_path):
    config = _make_ns_config(
        tmp_path,
        initial_condition=vector_expr(
            x=scalar_expr(
                "gaussian_bump",
                sigma=0.2,
                amplitude=0.5,
                center=[0.5, 0.5],
            ),
            y=constant(0.0),
        ),
    )
    result, output_dir = _run_preset(config)
    assert result.solver_converged is True

    vx = np.load(output_dir / "velocity_x.npy")
    assert vx.shape == (2, *config.output.resolution)
    assert np.max(np.abs(vx[0])) > 0


def test_navier_stokes_body_force(tmp_path):
    config = _make_ns_config(
        tmp_path,
        initial_condition=scalar_expr("custom"),
        source=vector_expr(x=constant(0.0), y=constant(-1.0)),
    )
    result, output_dir = _run_preset(config)
    assert result.solver_converged is True

    vy = np.load(output_dir / "velocity_y.npy")
    assert np.max(np.abs(vy)) > 0


def test_cahn_hilliard_constant_ic(tmp_path):
    config = SimulationConfig(
        preset="cahn_hilliard",
        parameters={
            "lmbda": 0.01,
            "barrier_height": 100.0,
            "mobility": 1.0,
            "theta": 0.5,
        },
        domain=DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
        ),
        fields={
            "c": FieldConfig(
                initial_condition=constant(0.5),
                output=FieldOutputConfig(mode="scalar"),
            )
        },
        output=OutputConfig(
            path=tmp_path, resolution=[4, 4], num_frames=2, formats=["numpy"]
        ),
        solver=SolverConfig(
            options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "none",
                "ksp_type": "preonly",
                "pc_type": "lu",
            }
        ),
        time=TimeConfig(dt=5e-6, t_end=5e-6),
        seed=42,
    )

    result, output_dir = _run_preset(config)
    assert result.solver_converged is True

    arr = np.load(output_dir / "c.npy")
    assert arr.shape == (2, *config.output.resolution)
    assert np.allclose(arr[0], 0.5, atol=0.05)

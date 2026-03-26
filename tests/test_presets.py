"""Tests for running actual PDE preset simulations."""

import logging

import numpy as np
import pytest
import ufl
from dolfinx import fem

from plm_data.core.config import (
    BoundaryConditionConfig,
    DomainConfig,
    FieldExpressionConfig,
    InputConfig,
    OutputConfig,
    OutputSelectionConfig,
    SimulationConfig,
    SolverConfig,
    TimeConfig,
)
from plm_data.core.mesh import create_domain
from plm_data.core.output import FrameWriter
from plm_data.core.runtime import is_complex_runtime
from plm_data.presets import get_preset
from plm_data.presets.base import StationaryLinearProblem
from plm_data.presets.basic.helmholtz import _check_resonance
from plm_data.presets.metadata import (
    OutputSpec,
    PresetSpec,
    StateSpec,
)


def constant(value):
    return FieldExpressionConfig(type="constant", params={"value": value})


def scalar_expr(expr_type: str, **params):
    return FieldExpressionConfig(type=expr_type, params=params)


def vector_expr(**components):
    return FieldExpressionConfig(components=components)


def vector_zero():
    return FieldExpressionConfig(type="zero", params={})


def output_fields(**modes):
    return {name: OutputSelectionConfig(mode=mode) for name, mode in modes.items()}


_VECTOR_STATIONARY_SPEC = PresetSpec(
    name="vector_dummy",
    category="tests",
    description="Dummy vector-valued stationary problem for engine tests",
    equations={"u": "u = 0"},
    parameters=[],
    inputs={},
    states={"u": StateSpec(name="u", shape="vector")},
    outputs={
        "u": OutputSpec(
            name="u",
            shape="vector",
            output_mode="components",
            source_name="u",
        )
    },
    steady_state=True,
    supported_dimensions=[2],
)


class _DummyVectorStationaryProblem(StationaryLinearProblem):
    def create_function_space(self, domain_geom):
        gdim = domain_geom.mesh.geometry.dim
        return fem.functionspace(domain_geom.mesh, ("Lagrange", 1, (gdim,)))

    def create_boundary_conditions(self, V, domain_geom):
        return []

    def create_forms(self, V, domain_geom):
        gdim = domain_geom.mesh.geometry.dim
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = ufl.inner(u, v) * ufl.dx
        L = (
            ufl.inner(
                fem.Constant(domain_geom.mesh, np.zeros(gdim, dtype=np.float64)), v
            )
            * ufl.dx
        )
        return a, L

    def export_solution_fields(self, solution):
        solution.name = "u"
        return {"u": solution}


def _run_preset(config):
    preset = get_preset(config.preset)
    output_dir = config.output.path / preset.spec.category / preset.spec.name
    writer = FrameWriter(output_dir, config, preset.spec)
    result = preset.build_problem(config).run(writer)
    writer.finalize()
    return result, output_dir


def _make_maxwell_pulse_config(tmp_path, *, gdim: int):
    if gdim == 2:
        domain = DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
        )
        center = [0.25, 0.5]
        boundary_conditions = {
            "x-": BoundaryConditionConfig(type="absorbing", value=vector_zero()),
            "x+": BoundaryConditionConfig(type="absorbing", value=vector_zero()),
            "y-": BoundaryConditionConfig(type="dirichlet", value=vector_zero()),
            "y+": BoundaryConditionConfig(type="dirichlet", value=vector_zero()),
        }
        source = vector_expr(
            x=scalar_expr("gaussian_bump", amplitude=1.0, sigma=0.06, center=center),
            y=constant(0.0),
        )
    else:
        domain = DomainConfig(
            type="box",
            params={"size": [1.0, 1.0, 1.0], "mesh_resolution": [4, 4, 4]},
        )
        center = [0.25, 0.5, 0.5]
        boundary_conditions = {
            "x-": BoundaryConditionConfig(type="absorbing", value=vector_zero()),
            "x+": BoundaryConditionConfig(type="absorbing", value=vector_zero()),
            "y-": BoundaryConditionConfig(type="absorbing", value=vector_zero()),
            "y+": BoundaryConditionConfig(type="absorbing", value=vector_zero()),
            "z-": BoundaryConditionConfig(type="dirichlet", value=vector_zero()),
            "z+": BoundaryConditionConfig(type="dirichlet", value=vector_zero()),
        }
        source = vector_expr(
            x=scalar_expr("gaussian_bump", amplitude=1.0, sigma=0.08, center=center),
            y=constant(0.0),
            z=constant(0.0),
        )

    return SimulationConfig(
        preset="maxwell_pulse",
        parameters={
            "epsilon_r": 1.0,
            "mu_r": 1.0,
            "sigma": 0.05,
            "pulse_amplitude": 3.0,
            "pulse_frequency": 6.0,
            "pulse_width": 0.08,
            "pulse_delay": 0.1,
        },
        domain=domain,
        inputs={
            "electric_field": InputConfig(
                boundary_conditions=boundary_conditions,
                source=source,
                initial_condition=vector_zero(),
            )
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4] * gdim,
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(electric_field="components"),
        ),
        solver=SolverConfig(
            options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            }
        ),
        time=TimeConfig(dt=0.02, t_end=0.02),
        seed=42,
    )


def _make_maxwell_config(tmp_path, *, gdim: int):
    if gdim == 2:
        domain = DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
        )
        center = [0.35, 0.5]
        boundary_names = ("x-", "x+", "y-", "y+")
        source = vector_expr(
            x=scalar_expr(
                "gaussian_bump",
                amplitude="param:source_amplitude",
                sigma=0.08,
                center=center,
            ),
            y=constant(0.0),
        )
    else:
        domain = DomainConfig(
            type="box",
            params={"size": [1.0, 1.0, 1.0], "mesh_resolution": [3, 3, 3]},
        )
        center = [0.35, 0.5, 0.5]
        boundary_names = ("x-", "x+", "y-", "y+", "z-", "z+")
        source = vector_expr(
            x=scalar_expr(
                "gaussian_bump",
                amplitude="param:source_amplitude",
                sigma=0.1,
                center=center,
            ),
            y=constant(0.0),
            z=constant(0.0),
        )

    return SimulationConfig(
        preset="maxwell",
        parameters={
            "epsilon_r": 1.0,
            "mu_r": 1.0,
            "k0": 8.0,
            "source_amplitude": 1.0,
        },
        domain=domain,
        inputs={
            "electric_field": InputConfig(
                boundary_conditions={
                    name: BoundaryConditionConfig(type="absorbing", value=vector_zero())
                    for name in boundary_names
                },
                source=source,
            )
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4] * gdim,
            num_frames=1,
            formats=["numpy"],
            fields=output_fields(electric_field="components"),
        ),
        solver=SolverConfig(
            options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            }
        ),
        seed=42,
    )


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


def test_helmholtz_preset(helmholtz_config):
    result, output_dir = _run_preset(helmholtz_config)
    assert result.solver_converged is True
    assert result.num_dofs > 0

    arr = np.load(output_dir / "u.npy")
    assert arr.shape == (1, *helmholtz_config.output.resolution)
    assert np.max(arr) > 0


def test_helmholtz_resonance_warning(caplog):
    """Warn when k^2 is near a Laplacian eigenvalue."""
    with caplog.at_level(logging.WARNING):
        _check_resonance(4.44, [1.0, 1.0])
    assert "ill-conditioned" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _check_resonance(2.0, [1.0, 1.0])
    assert caplog.text == ""


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
        inputs={
            "velocity": InputConfig(
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
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4],
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(velocity="components", pressure="scalar"),
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


def test_stokes_preset(stokes_config):
    result, output_dir = _run_preset(stokes_config)
    assert result.solver_converged is True
    assert result.num_dofs > 0

    for field in ["velocity_x", "velocity_y", "pressure"]:
        arr = np.load(output_dir / f"{field}.npy")
        assert arr.shape == (1, *stokes_config.output.resolution)

    vx = np.load(output_dir / "velocity_x.npy")
    assert np.max(np.abs(vx)) > 0


def _make_stokes_config(tmp_path, *, source=None, parameters=None):
    domain = DomainConfig(
        type="rectangle",
        params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
    )
    if parameters is None:
        parameters = {"nu": 1.0}
    if source is None:
        source = scalar_expr("none")

    return SimulationConfig(
        preset="stokes",
        parameters=parameters,
        domain=domain,
        inputs={
            "velocity": InputConfig(
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
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4],
            num_frames=1,
            formats=["numpy"],
            fields=output_fields(velocity="components", pressure="scalar"),
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
        seed=42,
    )


def test_stokes_body_force(tmp_path):
    config = _make_stokes_config(
        tmp_path,
        source=vector_expr(x=constant(0.0), y=constant(-1.0)),
    )
    result, output_dir = _run_preset(config)
    assert result.solver_converged is True

    vy = np.load(output_dir / "velocity_y.npy")
    assert np.max(np.abs(vy)) > 0


def test_stokes_3d(tmp_path):
    config = SimulationConfig(
        preset="stokes",
        parameters={"nu": 1.0},
        domain=DomainConfig(
            type="box",
            params={"size": [1.0, 1.0, 1.0], "mesh_resolution": [4, 4, 4]},
        ),
        inputs={
            "velocity": InputConfig(
                boundary_conditions={
                    "x-": BoundaryConditionConfig(
                        type="dirichlet",
                        value=vector_expr(
                            x=constant(0.0), y=constant(0.0), z=constant(0.0)
                        ),
                    ),
                    "x+": BoundaryConditionConfig(
                        type="dirichlet",
                        value=vector_expr(
                            x=constant(0.0), y=constant(0.0), z=constant(0.0)
                        ),
                    ),
                    "y-": BoundaryConditionConfig(
                        type="dirichlet",
                        value=vector_expr(
                            x=constant(0.0), y=constant(0.0), z=constant(0.0)
                        ),
                    ),
                    "y+": BoundaryConditionConfig(
                        type="dirichlet",
                        value=vector_expr(
                            x=constant(1.0), y=constant(0.0), z=constant(0.0)
                        ),
                    ),
                    "z-": BoundaryConditionConfig(
                        type="dirichlet",
                        value=vector_expr(
                            x=constant(0.0), y=constant(0.0), z=constant(0.0)
                        ),
                    ),
                    "z+": BoundaryConditionConfig(
                        type="dirichlet",
                        value=vector_expr(
                            x=constant(0.0), y=constant(0.0), z=constant(0.0)
                        ),
                    ),
                },
                source=scalar_expr("none"),
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4, 4],
            num_frames=1,
            formats=["numpy"],
            fields=output_fields(velocity="components", pressure="scalar"),
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
        seed=42,
    )
    result, output_dir = _run_preset(config)
    assert result.solver_converged is True
    assert result.num_dofs > 0

    for field in ["velocity_x", "velocity_y", "velocity_z", "pressure"]:
        arr = np.load(output_dir / f"{field}.npy")
        assert arr.shape == (1, *config.output.resolution)

    vx = np.load(output_dir / "velocity_x.npy")
    assert np.max(np.abs(vx)) > 0


def test_stokes_hidden_pressure(tmp_path):
    config = _make_stokes_config(tmp_path)
    config.output.fields["pressure"] = OutputSelectionConfig(mode="hidden")
    result, output_dir = _run_preset(config)
    assert result.solver_converged is True

    assert (output_dir / "velocity_x.npy").exists()
    assert (output_dir / "velocity_y.npy").exists()
    assert not (output_dir / "pressure.npy").exists()


def test_stationary_linear_problem_counts_vector_dofs(tmp_path, direct_solver):
    config = SimulationConfig(
        preset="vector_dummy",
        parameters={},
        domain=DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [4, 4]},
        ),
        inputs={},
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4],
            num_frames=1,
            formats=["numpy"],
            fields=output_fields(u="components"),
        ),
        solver=direct_solver,
        seed=42,
    )

    output_dir = (
        config.output.path
        / _VECTOR_STATIONARY_SPEC.category
        / _VECTOR_STATIONARY_SPEC.name
    )
    writer = FrameWriter(output_dir, config, _VECTOR_STATIONARY_SPEC)
    result = _DummyVectorStationaryProblem(_VECTOR_STATIONARY_SPEC, config).run(writer)
    writer.finalize()

    domain_geom = create_domain(config.domain)
    V = fem.functionspace(domain_geom.mesh, ("Lagrange", 1, (2,)))
    expected_num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    assert result.num_dofs == expected_num_dofs


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
        inputs={
            "c": InputConfig(
                initial_condition=constant(0.5),
            )
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4],
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(c="scalar"),
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


def test_maxwell_pulse_preset_2d_single_step(tmp_path):
    config = _make_maxwell_pulse_config(tmp_path, gdim=2)
    result, output_dir = _run_preset(config)
    assert result.solver_converged is True
    assert result.num_timesteps == 1

    ex = np.load(output_dir / "electric_field_x.npy")
    ey = np.load(output_dir / "electric_field_y.npy")
    assert ex.shape == (2, *config.output.resolution)
    assert ey.shape == (2, *config.output.resolution)
    assert np.max(np.abs(ex)) > 0


def test_maxwell_pulse_preset_3d_single_step(tmp_path):
    config = _make_maxwell_pulse_config(tmp_path, gdim=3)
    result, output_dir = _run_preset(config)
    assert result.solver_converged is True
    assert result.num_timesteps == 1

    for field in ["electric_field_x", "electric_field_y", "electric_field_z"]:
        arr = np.load(output_dir / f"{field}.npy")
        assert arr.shape == (2, *config.output.resolution)


def test_maxwell_preset_requires_complex_runtime(tmp_path):
    if is_complex_runtime():
        pytest.skip("runtime is already complex-capable")
    config = _make_maxwell_config(tmp_path, gdim=2)
    preset = get_preset(config.preset)
    with pytest.raises(RuntimeError, match="complex-valued DOLFINx/PETSc build"):
        preset.build_problem(config)


@pytest.mark.skipif(
    not is_complex_runtime(),
    reason="harmonic Maxwell requires a complex-valued runtime",
)
def test_maxwell_preset_2d(tmp_path):
    config = _make_maxwell_config(tmp_path, gdim=2)
    result, output_dir = _run_preset(config)
    assert result.solver_converged is True

    ex = np.load(output_dir / "electric_field_x.npy")
    ey = np.load(output_dir / "electric_field_y.npy")
    assert ex.shape == (1, *config.output.resolution)
    assert ey.shape == (1, *config.output.resolution)
    assert np.iscomplexobj(ex)


@pytest.mark.skipif(
    not is_complex_runtime(),
    reason="harmonic Maxwell requires a complex-valued runtime",
)
def test_maxwell_preset_3d(tmp_path):
    config = _make_maxwell_config(tmp_path, gdim=3)
    result, output_dir = _run_preset(config)
    assert result.solver_converged is True

    for field in ["electric_field_x", "electric_field_y", "electric_field_z"]:
        arr = np.load(output_dir / f"{field}.npy")
        assert arr.shape == (1, *config.output.resolution)
        assert np.iscomplexobj(arr)

"""Tests for running actual PDE preset simulations."""

import importlib.util
import logging
from dataclasses import replace

import numpy as np
import pytest
import ufl
from dolfinx import fem

import plm_data.core.periodic as periodic_mod
from plm_data.core.config import (
    BoundaryConditionConfig,
    BoundaryFieldConfig,
    DomainConfig,
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
from plm_data.presets.base import StationaryLinearProblem, TransientLinearProblem
from plm_data.presets.basic.helmholtz import _check_resonance
from plm_data.presets.metadata import (
    OutputSpec,
    PresetSpec,
    StateSpec,
)
from tests.preset_matrix import (
    boundary_field_config,
    constant,
    output_fields,
    run_preset,
    scalar_expr,
    vector_expr,
    vector_zero,
)

HAS_DOLFINX_MPC = importlib.util.find_spec("dolfinx_mpc") is not None


_VECTOR_STATIONARY_SPEC = PresetSpec(
    name="vector_dummy",
    category="tests",
    description="Dummy vector-valued stationary problem for engine tests",
    equations={"u": "u = 0"},
    parameters=[],
    inputs={},
    boundary_fields={},
    states={"u": StateSpec(name="u", shape="vector")},
    outputs={
        "u": OutputSpec(
            name="u",
            shape="vector",
            output_mode="components",
            source_name="u",
        )
    },
    static_fields=[],
    steady_state=True,
    supported_dimensions=[2],
)

_SCALAR_TRANSIENT_SPEC = PresetSpec(
    name="scalar_transient_dummy",
    category="tests",
    description="Dummy scalar transient problem for engine tests",
    equations={"u": "du/dt = 0"},
    parameters=[],
    inputs={},
    boundary_fields={},
    states={"u": StateSpec(name="u", shape="scalar")},
    outputs={
        "u": OutputSpec(
            name="u",
            shape="scalar",
            output_mode="scalar",
            source_name="u",
        )
    },
    static_fields=[],
    steady_state=False,
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


class _DummyScalarTransientProblem(TransientLinearProblem):
    def setup(self) -> None:
        self._step_calls = 0
        domain_geom = create_domain(self.config.domain)
        self._V = fem.functionspace(domain_geom.mesh, ("Lagrange", 1))
        self._u = fem.Function(self._V, name="u")

    def step(self, t: float, dt: float) -> bool:
        self._step_calls += 1
        self._u.x.array[:] = float(self._step_calls)
        self._u.x.scatter_forward()
        return True

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"u": self._u}

    def get_num_dofs(self) -> int:
        return self._V.dofmap.index_map.size_global * self._V.dofmap.index_map_bs


_run_preset = run_preset


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
                source=source,
                initial_condition=vector_zero(),
            )
        },
        boundary_conditions={
            "electric_field": boundary_field_config(boundary_conditions)
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
                source=source,
            )
        },
        boundary_conditions={
            "electric_field": boundary_field_config(
                {
                    name: BoundaryConditionConfig(type="absorbing", value=vector_zero())
                    for name in boundary_names
                }
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


def test_helmholtz_resonance_warning(caplog):
    """Warn when k^2 is near a Laplacian eigenvalue."""
    with caplog.at_level(logging.WARNING):
        _check_resonance(4.44, [1.0, 1.0])
    assert "ill-conditioned" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _check_resonance(2.0, [1.0, 1.0])
    assert caplog.text == ""


def test_preset_spec_rejects_unknown_static_fields():
    with pytest.raises(ValueError, match="static field 'missing'"):
        replace(_SCALAR_TRANSIENT_SPEC, static_fields=["missing"])


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
                source=source,
                initial_condition=initial_condition,
            ),
        },
        boundary_conditions={
            "velocity": boundary_field_config(
                {
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
                }
            )
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
                source=source,
            ),
        },
        boundary_conditions={
            "velocity": boundary_field_config(
                {
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
                }
            )
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


def _make_thermal_convection_config(tmp_path, *, gdim: int):
    if gdim == 2:
        domain = DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [6, 6]},
        )
        periodic_axes = (0,)
        velocity_boundary_conditions = {
            "y-": BoundaryConditionConfig(
                type="dirichlet",
                value=vector_expr(x=constant(0.0), y=constant(0.0)),
            ),
            "y+": BoundaryConditionConfig(
                type="dirichlet",
                value=vector_expr(x=constant(0.0), y=constant(0.0)),
            ),
        }
        temperature_boundary_conditions = {
            "y-": BoundaryConditionConfig(type="dirichlet", value=constant(1.0)),
            "y+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
        }
    else:
        domain = DomainConfig(
            type="box",
            params={"size": [1.0, 1.0, 1.0], "mesh_resolution": [4, 4, 3]},
        )
        periodic_axes = (0, 1)
        velocity_boundary_conditions = {
            "z-": BoundaryConditionConfig(
                type="dirichlet",
                value=vector_expr(x=constant(0.0), y=constant(0.0), z=constant(0.0)),
            ),
            "z+": BoundaryConditionConfig(
                type="dirichlet",
                value=vector_expr(x=constant(0.0), y=constant(0.0), z=constant(0.0)),
            ),
        }
        temperature_boundary_conditions = {
            "z-": BoundaryConditionConfig(type="dirichlet", value=constant(1.0)),
            "z+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
        }

    return SimulationConfig(
        preset="thermal_convection",
        parameters={"Ra": 1000.0, "Pr": 1.0, "k": 1.0},
        domain=domain,
        inputs={
            "velocity": InputConfig(
                source=scalar_expr("none"),
                initial_condition=vector_zero(),
            ),
            "temperature": InputConfig(
                source=scalar_expr("none"),
                initial_condition=scalar_expr(
                    "random_perturbation",
                    mean=0.5,
                    std=0.02,
                ),
            ),
        },
        boundary_conditions={
            "velocity": boundary_field_config(
                velocity_boundary_conditions,
                periodic_axes=periodic_axes,
            ),
            "temperature": boundary_field_config(
                temperature_boundary_conditions,
                periodic_axes=periodic_axes,
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4] * gdim,
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(
                velocity="components",
                pressure="scalar",
                temperature="scalar",
            ),
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
        time=TimeConfig(dt=0.05, t_end=0.05),
        seed=42,
    )


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
                source=scalar_expr("none"),
            ),
        },
        boundary_conditions={
            "velocity": boundary_field_config(
                {
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
                }
            )
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
        boundary_conditions={},
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


def test_transient_problem_uses_exact_integer_step_count(tmp_path, direct_solver):
    config = SimulationConfig(
        preset="scalar_transient_dummy",
        parameters={},
        domain=DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [2, 2]},
        ),
        inputs={},
        boundary_conditions={},
        output=OutputConfig(
            path=tmp_path,
            resolution=[2, 2],
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(u="scalar"),
        ),
        solver=direct_solver,
        time=TimeConfig(dt=0.01, t_end=5.0),
        seed=42,
    )

    output_dir = (
        config.output.path
        / _SCALAR_TRANSIENT_SPEC.category
        / _SCALAR_TRANSIENT_SPEC.name
    )
    writer = FrameWriter(output_dir, config, _SCALAR_TRANSIENT_SPEC)
    result = _DummyScalarTransientProblem(_SCALAR_TRANSIENT_SPEC, config).run(writer)
    writer.finalize()

    assert result.solver_converged is True
    assert result.num_timesteps == 500


@pytest.mark.skipif(
    not HAS_DOLFINX_MPC,
    reason="periodic heat solve requires dolfinx_mpc",
)
def test_heat_periodic_domain_single_step(tmp_path, direct_solver):
    config = SimulationConfig(
        preset="heat",
        parameters={},
        domain=DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
        ),
        inputs={
            "u": InputConfig(
                source=scalar_expr("none"),
                initial_condition=scalar_expr(
                    "gaussian_bump",
                    sigma=0.1,
                    amplitude=1.0,
                    center=[0.5, 0.5],
                ),
            )
        },
        boundary_conditions={"u": boundary_field_config({}, periodic_axes=(0, 1))},
        output=OutputConfig(
            path=tmp_path,
            resolution=[8, 8],
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(u="scalar"),
        ),
        solver=direct_solver,
        time=TimeConfig(dt=0.01, t_end=0.01),
        seed=42,
        coefficients={"kappa": constant(0.01)},
    )
    result, output_dir = _run_preset(config)
    assert result.solver_converged is True

    arr = np.load(output_dir / "u.npy")
    assert np.allclose(arr[-1, :, 0], arr[-1, :, -1], atol=1e-4)
    assert np.allclose(arr[-1, 0, :], arr[-1, -1, :], atol=1e-4)


def test_periodic_heat_requires_dolfinx_mpc(monkeypatch, tmp_path, direct_solver):
    original_import_module = periodic_mod.importlib.import_module

    def _missing_dolfinx_mpc(name, *args, **kwargs):
        if name == "dolfinx_mpc":
            raise ImportError("missing dolfinx_mpc")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(periodic_mod.importlib, "import_module", _missing_dolfinx_mpc)

    config = SimulationConfig(
        preset="heat",
        parameters={},
        domain=DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
        ),
        inputs={
            "u": InputConfig(
                source=scalar_expr("none"),
                initial_condition=scalar_expr(
                    "gaussian_bump",
                    sigma=0.1,
                    amplitude=1.0,
                    center=[0.5, 0.5],
                ),
            )
        },
        boundary_conditions={"u": boundary_field_config({}, periodic_axes=(0, 1))},
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4],
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(u="scalar"),
        ),
        solver=direct_solver,
        time=TimeConfig(dt=0.01, t_end=0.01),
        seed=42,
        coefficients={"kappa": constant(0.01)},
    )

    with pytest.raises(RuntimeError, match="dolfinx_mpc"):
        _run_preset(config)


def test_heat_manufactured_source_approaches_expected_profile(tmp_path, direct_solver):
    config = SimulationConfig(
        preset="heat",
        parameters={},
        domain=DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [48, 48]},
        ),
        inputs={
            "u": InputConfig(
                source=scalar_expr(
                    "cosine_product",
                    amplitude=1.9739208802178716,
                    kx=1.0,
                    ky=1.0,
                ),
                initial_condition=scalar_expr("constant", value=0.0),
            )
        },
        boundary_conditions={
            "u": boundary_field_config(
                {
                    "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                    "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                    "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                    "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                }
            )
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[32, 32],
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(u="scalar"),
        ),
        solver=direct_solver,
        time=TimeConfig(dt=0.02, t_end=2.0),
        seed=42,
        coefficients={"kappa": constant(0.1)},
    )

    result, output_dir = _run_preset(config)
    assert result.solver_converged is True

    arr = np.load(output_dir / "u.npy")[-1]
    x = np.linspace(0.0, 1.0, config.output.resolution[0])
    y = np.linspace(0.0, 1.0, config.output.resolution[1])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    expected = np.cos(np.pi * xx) * np.cos(np.pi * yy)
    assert np.max(np.abs(arr - expected)) < 0.12


def test_heat_variable_diffusivity_evolves_nontrivially(tmp_path, direct_solver):
    config = SimulationConfig(
        preset="heat",
        parameters={},
        domain=DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [40, 40]},
        ),
        inputs={
            "u": InputConfig(
                source=scalar_expr("none"),
                initial_condition=scalar_expr("constant", value=1.0),
            )
        },
        boundary_conditions={
            "u": boundary_field_config(
                {
                    "x-": BoundaryConditionConfig(
                        type="dirichlet", value=constant(0.0)
                    ),
                    "x+": BoundaryConditionConfig(
                        type="dirichlet", value=constant(0.0)
                    ),
                    "y-": BoundaryConditionConfig(
                        type="dirichlet", value=constant(0.0)
                    ),
                    "y+": BoundaryConditionConfig(
                        type="dirichlet", value=constant(0.0)
                    ),
                }
            )
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[24, 24],
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(u="scalar"),
        ),
        solver=direct_solver,
        time=TimeConfig(dt=0.005, t_end=0.2),
        seed=42,
        coefficients={
            "kappa": scalar_expr(
                "radial_cosine",
                base=0.05,
                amplitude=0.03,
                frequency=12.566370614359172,
                center=[0.5, 0.5],
            )
        },
    )

    result, output_dir = _run_preset(config)
    assert result.solver_converged is True

    arr = np.load(output_dir / "u.npy")
    assert np.all(np.isfinite(arr))
    assert np.max(arr[-1]) < np.max(arr[0])
    assert np.std(arr[-1]) > 1e-3


@pytest.mark.skipif(
    not HAS_DOLFINX_MPC,
    reason="periodic Navier-Stokes solve requires dolfinx_mpc",
)
def test_navier_stokes_fully_periodic_domain(tmp_path):
    config = _make_ns_config(
        tmp_path,
        initial_condition=vector_expr(
            x=scalar_expr(
                "gaussian_bump",
                sigma=0.15,
                amplitude=0.5,
                center=[0.5, 0.5],
            ),
            y=constant(0.0),
        ),
        parameters={"Re": 10.0, "k": 1.0},
    )
    config.boundary_conditions["velocity"] = boundary_field_config(
        {},
        periodic_axes=(0, 1),
    )
    config.output.resolution = [6, 6]

    result, output_dir = _run_preset(config)
    assert result.solver_converged is True

    vx = np.load(output_dir / "velocity_x.npy")
    vy = np.load(output_dir / "velocity_y.npy")
    pressure = np.load(output_dir / "pressure.npy")
    assert np.allclose(vx[-1, 0, :], vx[-1, -1, :], atol=1e-4)
    assert np.allclose(vx[-1, :, 0], vx[-1, :, -1], atol=1e-4)
    assert np.allclose(vy[-1, 0, :], vy[-1, -1, :], atol=1e-4)
    assert np.allclose(vy[-1, :, 0], vy[-1, :, -1], atol=1e-4)
    assert np.allclose(pressure[-1, 0, :], pressure[-1, -1, :], atol=1e-4)
    assert np.allclose(pressure[-1, :, 0], pressure[-1, :, -1], atol=1e-4)


def test_thermal_convection_rejects_mismatched_periodic_fields(tmp_path):
    config = _make_thermal_convection_config(tmp_path, gdim=2)
    config.boundary_conditions["temperature"] = boundary_field_config(
        {
            "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "y-": BoundaryConditionConfig(type="dirichlet", value=constant(1.0)),
            "y+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
        }
    )

    preset = get_preset(config.preset)
    problem = preset.build_problem(config)
    with pytest.raises(ValueError, match="identical periodic side pairs"):
        problem.load_domain_geometry()


def test_thermal_convection_requires_standard_boundary_names(tmp_path):
    config = _make_thermal_convection_config(tmp_path, gdim=2)
    config.boundary_conditions["velocity"] = BoundaryFieldConfig(
        sides={
            "left": [
                BoundaryConditionConfig(
                    type="dirichlet",
                    value=vector_expr(x=constant(0.0), y=constant(0.0)),
                )
            ],
            "right": [
                BoundaryConditionConfig(
                    type="dirichlet",
                    value=vector_expr(x=constant(0.0), y=constant(0.0)),
                )
            ],
            "bottom": [
                BoundaryConditionConfig(
                    type="dirichlet",
                    value=vector_expr(x=constant(0.0), y=constant(0.0)),
                )
            ],
            "top": [
                BoundaryConditionConfig(
                    type="dirichlet",
                    value=vector_expr(x=constant(0.0), y=constant(0.0)),
                )
            ],
        }
    )
    config.boundary_conditions["temperature"] = BoundaryFieldConfig(
        sides={
            "left": [BoundaryConditionConfig(type="neumann", value=constant(0.0))],
            "right": [BoundaryConditionConfig(type="neumann", value=constant(0.0))],
            "bottom": [BoundaryConditionConfig(type="dirichlet", value=constant(1.0))],
            "top": [BoundaryConditionConfig(type="dirichlet", value=constant(0.0))],
        }
    )

    preset = get_preset(config.preset)
    problem = preset.build_problem(config)
    domain_geom = create_domain(config.domain)
    domain_geom.boundary_names = {"left": 1, "right": 2, "bottom": 3, "top": 4}

    with pytest.raises(ValueError, match="requires standard boundary names"):
        problem.validate_boundary_conditions(domain_geom)


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
def test_maxwell_preset_3d(tmp_path):
    config = _make_maxwell_config(tmp_path, gdim=3)
    result, output_dir = _run_preset(config)
    assert result.solver_converged is True

    for field in ["electric_field_x", "electric_field_y", "electric_field_z"]:
        arr = np.load(output_dir / f"{field}.npy")
        assert arr.shape == (1, *config.output.resolution)
        assert np.iscomplexobj(arr)

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
    PeriodicMapConfig,
    SimulationConfig,
    TimeConfig,
)
from plm_data.core.mesh import create_domain
from plm_data.core.output import FrameWriter
from plm_data.core.runtime import is_complex_runtime
from plm_data.core.solver_strategies import (
    CONSTANT_LHS_CURL_DIRECT,
    CONSTANT_LHS_SCALAR_NONSYMMETRIC,
    CONSTANT_LHS_SCALAR_SPD,
    STATIONARY_INDEFINITE_DIRECT,
    STEADY_SADDLE_POINT,
    TRANSIENT_MIXED_DIRECT,
    TRANSIENT_SADDLE_POINT,
)
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
    cahn_hilliard_solver_config,
    constant,
    direct_solver_config,
    make_advection_config,
    make_burgers_config,
    make_gray_scott_config,
    flow_solver_config,
    make_shallow_water_config,
    make_swift_hohenberg_config,
    make_superlattice_config,
    make_van_der_pol_config,
    make_wave_config,
    nonlinear_mixed_direct_solver_config,
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
        solver=direct_solver_config(CONSTANT_LHS_CURL_DIRECT),
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
        solver=direct_solver_config(STATIONARY_INDEFINITE_DIRECT),
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
        solver=flow_solver_config(TRANSIENT_SADDLE_POINT),
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
        solver=flow_solver_config(STEADY_SADDLE_POINT),
        seed=42,
    )


def _burgers_periodic_ic(*, gdim: int):
    if gdim == 2:
        return vector_expr(
            x=scalar_expr("sine_product", amplitude=1.0, ky=2.0),
            y=scalar_expr("sine_product", amplitude=-1.0, kx=2.0),
        )
    return vector_expr(
        x=scalar_expr("sine_product", amplitude=0.9, ky=2.0),
        y=scalar_expr("sine_product", amplitude=0.9, kz=2.0),
        z=scalar_expr("sine_product", amplitude=-0.9, kx=2.0),
    )


def _mixed_burgers_boundary_conditions():
    return {
        "x-": BoundaryConditionConfig(
            type="dirichlet",
            value=vector_expr(x=constant(0.0), y=constant(0.0)),
        ),
        "x+": BoundaryConditionConfig(
            type="neumann",
            value=vector_expr(x=constant(0.15), y=constant(-0.05)),
        ),
        "y-": BoundaryConditionConfig(
            type="neumann",
            value=vector_expr(x=constant(0.0), y=constant(0.0)),
        ),
        "y+": BoundaryConditionConfig(
            type="dirichlet",
            value=vector_expr(x=constant(0.0), y=constant(0.0)),
        ),
    }


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
                    "gaussian_noise",
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
        solver=flow_solver_config(TRANSIENT_MIXED_DIRECT),
        time=TimeConfig(dt=0.05, t_end=0.05),
        seed=42,
    )


def _make_shallow_water_runtime_config(tmp_path):
    return make_shallow_water_config(
        tmp_path,
        parameters={
            "gravity": 1.0,
            "mean_depth": 1.0,
            "drag": 0.01,
            "viscosity": 0.002,
            "coriolis": 0.5,
        },
        bathymetry=constant(0.0),
        initial_height=scalar_expr(
            "gaussian_bump",
            amplitude=0.15,
            sigma=0.06,
            center=[0.31, 0.43],
        ),
        initial_velocity=vector_zero(),
        height_boundary_conditions={},
        velocity_boundary_conditions={},
        height_periodic_axes=(0, 1),
        velocity_periodic_axes=(0, 1),
        mesh_resolution=(40, 40),
        output_resolution=(24, 24),
        time=TimeConfig(dt=0.01, t_end=0.4),
    )


def _make_keller_segel_config(
    tmp_path,
    *,
    rho_boundary_conditions,
    c_boundary_conditions,
    rho_periodic_axes=(),
    c_periodic_axes=(),
):
    return SimulationConfig(
        preset="keller_segel",
        parameters={
            "D_rho": 1.0,
            "D_c": 0.5,
            "chi0": 10.0,
            "alpha": 1.0,
            "beta": 1.0,
            "r": 1.0,
        },
        domain=DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [6, 6]},
        ),
        inputs={
            "rho": InputConfig(initial_condition=constant(1.0)),
            "c": InputConfig(initial_condition=constant(1.0)),
        },
        boundary_conditions={
            "rho": boundary_field_config(
                rho_boundary_conditions,
                periodic_axes=rho_periodic_axes,
            ),
            "c": boundary_field_config(
                c_boundary_conditions,
                periodic_axes=c_periodic_axes,
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4],
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(rho="scalar", c="scalar"),
        ),
        solver=direct_solver_config(CONSTANT_LHS_SCALAR_SPD),
        time=TimeConfig(dt=0.02, t_end=0.02),
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
        solver=flow_solver_config(STEADY_SADDLE_POINT),
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


@pytest.mark.skipif(
    not HAS_DOLFINX_MPC,
    reason="periodic advection solve requires dolfinx_mpc",
)
def test_advection_periodic_pure_advection_single_step(tmp_path):
    config = make_advection_config(
        tmp_path,
        gdim=2,
        velocity=vector_expr(
            x=constant(1.0),
            y=constant(0.35),
        ),
        diffusivity=scalar_expr("zero"),
        boundary_conditions={},
        source=scalar_expr("none"),
        initial_condition=scalar_expr(
            "gaussian_bump",
            sigma=0.08,
            amplitude=1.0,
            center=[0.25, 0.4],
        ),
        periodic_axes=(0, 1),
        mesh_resolution=(16, 16),
        output_resolution=(8, 8),
        time=TimeConfig(dt=0.01, t_end=0.01),
    )

    result, output_dir = _run_preset(config)
    assert result.solver_converged is True

    arr = np.load(output_dir / "u.npy")
    assert np.allclose(arr[-1, :, 0], arr[-1, :, -1], atol=1e-4)
    assert np.allclose(arr[-1, 0, :], arr[-1, -1, :], atol=1e-4)


def test_advection_rejects_spd_solver_strategy(tmp_path):
    config = make_advection_config(
        tmp_path,
        gdim=2,
        velocity=vector_expr(
            x=constant(1.0),
            y=constant(0.0),
        ),
        diffusivity=constant(0.01),
        boundary_conditions={
            "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
        },
        source=scalar_expr("none"),
        initial_condition=scalar_expr("constant", value=0.0),
        solver=direct_solver_config(CONSTANT_LHS_SCALAR_SPD),
        time=TimeConfig(dt=0.01, t_end=0.01),
    )

    with pytest.raises(ValueError, match="does not support solver strategy"):
        get_preset("advection").build_problem(config)


def test_gray_scott_rejects_spd_solver_strategy_with_advection(tmp_path):
    config = make_gray_scott_config(
        tmp_path,
        gdim=2,
        velocity=vector_expr(x=constant(1.0), y=constant(0.0)),
        u_initial_condition=scalar_expr(
            "gaussian_blobs",
            background=1.0,
            blobs=[{"amplitude": -0.5, "sigma": 0.08, "center": [0.5, 0.5]}],
        ),
        v_initial_condition=scalar_expr(
            "gaussian_blobs",
            background=0.0,
            blobs=[{"amplitude": 0.25, "sigma": 0.08, "center": [0.5, 0.5]}],
        ),
        u_boundary_conditions={},
        v_boundary_conditions={},
        u_periodic_axes=(0, 1),
        v_periodic_axes=(0, 1),
        solver=direct_solver_config(CONSTANT_LHS_SCALAR_SPD),
        time=TimeConfig(dt=1.0, t_end=1.0),
    )

    with pytest.raises(ValueError, match="requires solver strategy"):
        get_preset("gray_scott").build_problem(config)


def test_gray_scott_accepts_spd_solver_strategy_with_zero_velocity(tmp_path):
    config = make_gray_scott_config(
        tmp_path,
        gdim=2,
        velocity=vector_zero(),
        u_initial_condition=scalar_expr(
            "gaussian_blobs",
            background=1.0,
            blobs=[{"amplitude": -0.5, "sigma": 0.08, "center": [0.5, 0.5]}],
        ),
        v_initial_condition=scalar_expr(
            "gaussian_blobs",
            background=0.0,
            blobs=[{"amplitude": 0.25, "sigma": 0.08, "center": [0.5, 0.5]}],
        ),
        u_boundary_conditions={},
        v_boundary_conditions={},
        u_periodic_axes=(0, 1),
        v_periodic_axes=(0, 1),
        solver=direct_solver_config(CONSTANT_LHS_SCALAR_SPD),
        time=TimeConfig(dt=1.0, t_end=1.0),
    )

    get_preset("gray_scott").build_problem(config)


def test_swift_hohenberg_rejects_mixed_boundary_modes(tmp_path):
    config = make_swift_hohenberg_config(
        tmp_path,
        gdim=2,
        initial_condition=scalar_expr(
            "gaussian_bump",
            amplitude=1.0,
            sigma=0.12,
            center=[0.5, 0.5],
        ),
        velocity=vector_expr(x=constant(0.4), y=constant(0.0)),
        boundary_conditions={
            "y-": BoundaryConditionConfig(type="simply_supported"),
            "y+": BoundaryConditionConfig(type="simply_supported"),
        },
        periodic_axes=(0,),
        mesh_resolution=(8, 8),
        output_resolution=(4, 4),
        time=TimeConfig(dt=0.05, t_end=0.05),
    )

    with pytest.raises(ValueError, match="global boundary mode"):
        _run_preset(config)


def test_burgers_rejects_transient_saddle_point_strategy(tmp_path):
    config = make_burgers_config(
        tmp_path,
        gdim=2,
        parameters={"nu": 0.01},
        boundary_conditions=_mixed_burgers_boundary_conditions(),
        source=scalar_expr("none"),
        initial_condition=vector_expr(
            x=scalar_expr(
                "gaussian_bump",
                amplitude=0.8,
                sigma=0.12,
                center=[0.35, 0.45],
            ),
            y=scalar_expr("sine_product", amplitude=0.2, kx=1.0, ky=1.0),
        ),
        time=TimeConfig(dt=0.02, t_end=0.02),
        solver=flow_solver_config(TRANSIENT_SADDLE_POINT),
    )

    with pytest.raises(ValueError, match="does not support solver strategy"):
        get_preset("burgers").build_problem(config)


def test_advection_affine_velocity_single_step(tmp_path):
    config = make_advection_config(
        tmp_path,
        gdim=2,
        velocity=vector_expr(
            x=scalar_expr("affine", constant=-0.5, y=1.0),
            y=scalar_expr("affine", constant=0.5, x=-1.0),
        ),
        diffusivity=constant(0.005),
        boundary_conditions={
            "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
        },
        source=scalar_expr("none"),
        initial_condition=scalar_expr(
            "gaussian_bump",
            sigma=0.08,
            amplitude=1.0,
            center=[0.7, 0.5],
        ),
        mesh_resolution=(20, 20),
        output_resolution=(8, 8),
        time=TimeConfig(dt=0.01, t_end=0.01),
        solver=direct_solver_config(CONSTANT_LHS_SCALAR_NONSYMMETRIC),
    )

    result, output_dir = _run_preset(config)
    assert result.solver_converged is True
    arr = np.load(output_dir / "u.npy")
    assert np.max(np.abs(arr[-1] - arr[0])) > 1e-6


def test_burgers_mixed_vector_boundary_conditions_single_step(tmp_path):
    config = make_burgers_config(
        tmp_path,
        gdim=2,
        parameters={"nu": 0.02},
        boundary_conditions=_mixed_burgers_boundary_conditions(),
        source=scalar_expr("none"),
        initial_condition=vector_expr(
            x=scalar_expr(
                "gaussian_bump",
                amplitude=0.8,
                sigma=0.12,
                center=[0.35, 0.45],
            ),
            y=scalar_expr("sine_product", amplitude=0.2, kx=1.0, ky=1.0),
        ),
        mesh_resolution=(16, 16),
        output_resolution=(8, 8),
        time=TimeConfig(dt=0.02, t_end=0.02),
    )

    result, output_dir = _run_preset(config)
    assert result.solver_converged is True
    assert result.num_timesteps == 1

    velocity_x = np.load(output_dir / "velocity_x.npy")
    velocity_y = np.load(output_dir / "velocity_y.npy")
    assert np.max(np.abs(velocity_x[-1] - velocity_x[0])) > 1e-6
    assert np.max(np.abs(velocity_y[-1] - velocity_y[0])) > 1e-6


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


def test_wave_standing_mode_matches_expected_profile(tmp_path):
    config = make_wave_config(
        tmp_path,
        gdim=2,
        boundary_conditions={
            "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
        },
        initial_displacement=scalar_expr(
            "cosine_product",
            amplitude=1.0,
            kx=1,
            ky=1,
        ),
        initial_velocity=constant(0.0),
        forcing=scalar_expr("none"),
        mesh_resolution=(48, 48),
        output_resolution=(32, 32),
        time=TimeConfig(dt=0.005, t_end=0.1),
    )

    result, output_dir = _run_preset(config)
    assert result.solver_converged is True

    arr_u = np.load(output_dir / "u.npy")[-1]
    arr_v = np.load(output_dir / "v.npy")[-1]
    x = np.linspace(0.0, 1.0, config.output.resolution[0])
    y = np.linspace(0.0, 1.0, config.output.resolution[1])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mode = np.cos(np.pi * xx) * np.cos(np.pi * yy)
    omega = np.pi * np.sqrt(2.0)
    expected_u = np.cos(omega * config.time.t_end) * mode

    assert np.max(np.abs(arr_u - expected_u)) < 0.1
    assert np.max(np.abs(arr_v)) > 1e-2


def test_wave_preset_3d_single_step(tmp_path):
    config = make_wave_config(
        tmp_path,
        gdim=3,
        boundary_conditions={
            "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "z-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "z+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
        },
        initial_displacement=scalar_expr(
            "cosine_product",
            amplitude=1.0,
            kx=1,
            ky=1,
            kz=1,
        ),
        initial_velocity=constant(0.0),
        forcing=scalar_expr("none"),
        mesh_resolution=(4, 4, 4),
        output_resolution=(4, 4, 4),
        time=TimeConfig(dt=0.02, t_end=0.02),
    )

    result, output_dir = _run_preset(config)
    assert result.solver_converged is True
    assert result.num_timesteps == 1

    for field in ["u", "v"]:
        arr = np.load(output_dir / f"{field}.npy")
        assert arr.shape == (2, *config.output.resolution)
        assert np.all(np.isfinite(arr))


def test_wave_inhomogeneous_medium_2d_stays_nontrivial(tmp_path):
    config = make_wave_config(
        tmp_path,
        gdim=2,
        boundary_conditions={
            "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
        },
        initial_displacement=constant(0.0),
        initial_velocity=scalar_expr(
            "gaussian_bump",
            amplitude=12.0,
            sigma=0.06,
            center=[0.28, 0.44],
        ),
        forcing=scalar_expr("none"),
        mesh_resolution=(56, 56),
        output_resolution=(48, 48),
        damping=0.01,
        c_sq=scalar_expr(
            "radial_cosine",
            base=1.8,
            amplitude=0.65,
            frequency=18.0,
            center=[0.5, 0.5],
        ),
        time=TimeConfig(dt=0.004, t_end=0.8),
    )
    config.output.num_frames = 25

    result, output_dir = _run_preset(config)
    assert result.solver_converged is True
    assert result.num_timesteps == 200

    arr_u = np.load(output_dir / "u.npy")
    arr_v = np.load(output_dir / "v.npy")

    assert np.all(np.isfinite(arr_u))
    assert np.all(np.isfinite(arr_v))
    assert np.max(arr_u) > 0.15
    assert np.std(arr_u[-1]) > 1.0e-2
    assert np.std(arr_v[-1]) > 1.0e-2
    assert np.linalg.norm(arr_u[-1] - arr_u[len(arr_u) // 2]) > 1.0


def test_wave_inhomogeneous_medium_3d_short_run(tmp_path):
    config = make_wave_config(
        tmp_path,
        gdim=3,
        boundary_conditions={
            "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "z-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "z+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
        },
        initial_displacement=constant(0.0),
        initial_velocity=scalar_expr(
            "gaussian_bump",
            amplitude=8.0,
            sigma=0.08,
            center=[0.3, 0.42, 0.55],
        ),
        forcing=scalar_expr("none"),
        mesh_resolution=(16, 16, 16),
        output_resolution=(20, 20, 20),
        damping=0.01,
        c_sq=scalar_expr(
            "radial_cosine",
            base=1.8,
            amplitude=0.55,
            frequency=16.0,
            center=[0.5, 0.5, 0.5],
        ),
        time=TimeConfig(dt=0.01, t_end=0.25),
    )
    config.output.num_frames = 10

    result, output_dir = _run_preset(config)
    assert result.solver_converged is True
    assert result.num_timesteps == 25

    arr_u = np.load(output_dir / "u.npy")
    arr_v = np.load(output_dir / "v.npy")

    assert arr_u.shape == (10, *config.output.resolution)
    assert arr_v.shape == (10, *config.output.resolution)
    assert np.all(np.isfinite(arr_u))
    assert np.all(np.isfinite(arr_v))
    assert np.std(arr_u[-1]) > 5.0e-3
    assert np.std(arr_v[-1]) > 5.0e-3


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


@pytest.mark.skipif(
    not HAS_DOLFINX_MPC,
    reason="periodic Burgers solve requires dolfinx_mpc",
)
def test_burgers_fully_periodic_domain_3d(tmp_path):
    config = make_burgers_config(
        tmp_path,
        gdim=3,
        parameters={"nu": 0.01},
        boundary_conditions={},
        source=scalar_expr("none"),
        initial_condition=_burgers_periodic_ic(gdim=3),
        periodic_axes=(0, 1, 2),
        mesh_resolution=(6, 6, 6),
        output_resolution=(4, 4, 4),
        time=TimeConfig(dt=0.02, t_end=0.02),
        solver=cahn_hilliard_solver_config(),
    )

    result, output_dir = _run_preset(config)
    assert result.solver_converged is True

    for field_name in ("velocity_x", "velocity_y", "velocity_z"):
        arr = np.load(output_dir / f"{field_name}.npy")
        assert np.max(np.abs(arr[-1] - arr[0])) > 1e-6
        assert np.allclose(arr[-1, 0, :, :], arr[-1, -1, :, :], atol=1e-4)
        assert np.allclose(arr[-1, :, 0, :], arr[-1, :, -1, :], atol=1e-4)
        assert np.allclose(arr[-1, :, :, 0], arr[-1, :, :, -1], atol=1e-4)


@pytest.mark.skipif(
    not HAS_DOLFINX_MPC,
    reason="periodic superlattice solve requires dolfinx_mpc",
)
def test_superlattice_periodic_equilibrium_is_preserved(tmp_path):
    config = make_superlattice_config(
        tmp_path,
        gdim=2,
        u_1_initial_condition=constant(3.0),
        v_1_initial_condition=constant(3.0),
        u_2_initial_condition=constant(3.0),
        v_2_initial_condition=constant(10.0),
        u_1_boundary_conditions={},
        v_1_boundary_conditions={},
        u_2_boundary_conditions={},
        v_2_boundary_conditions={},
        u_1_periodic_axes=(0, 1),
        v_1_periodic_axes=(0, 1),
        u_2_periodic_axes=(0, 1),
        v_2_periodic_axes=(0, 1),
        mesh_resolution=(8, 8),
        output_resolution=(6, 6),
        time=TimeConfig(dt=0.01, t_end=0.01),
    )

    result, output_dir = _run_preset(config)
    assert result.solver_converged is True
    assert result.num_timesteps == 1

    expected_values = {
        "u_1": 3.0,
        "v_1": 3.0,
        "u_2": 3.0,
        "v_2": 10.0,
    }
    for field_name, expected in expected_values.items():
        arr = np.load(output_dir / f"{field_name}.npy")
        assert np.allclose(arr[0], expected, atol=1e-8)
        assert np.allclose(arr[-1], expected, atol=1e-8)


def test_van_der_pol_zero_equilibrium_is_preserved(tmp_path):
    config = make_van_der_pol_config(
        tmp_path,
        gdim=2,
        u_initial_condition=constant(0.0),
        v_initial_condition=constant(0.0),
        u_boundary_conditions={
            "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
        },
        v_boundary_conditions={
            "x-": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
            "x+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
            "y-": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
            "y+": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
        },
        mesh_resolution=(8, 8),
        output_resolution=(6, 6),
        time=TimeConfig(dt=0.01, t_end=0.01),
    )

    result, output_dir = _run_preset(config)
    assert result.solver_converged is True
    assert result.num_timesteps == 1

    for field_name in ("u", "v"):
        arr = np.load(output_dir / f"{field_name}.npy")
        assert np.allclose(arr[0], 0.0, atol=1e-10)
        assert np.allclose(arr[-1], 0.0, atol=1e-10)


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


@pytest.mark.skipif(
    not HAS_DOLFINX_MPC,
    reason="periodic shallow-water solve requires dolfinx_mpc",
)
def test_shallow_water_periodic_run_stays_nontrivial_and_positive(tmp_path):
    config = _make_shallow_water_runtime_config(tmp_path)
    config.output.num_frames = 9

    result, output_dir = _run_preset(config)
    assert result.solver_converged is True
    assert result.num_timesteps == 40

    height = np.load(output_dir / "height.npy")
    velocity_x = np.load(output_dir / "velocity_x.npy")
    velocity_y = np.load(output_dir / "velocity_y.npy")

    assert np.all(np.isfinite(height))
    assert np.all(np.isfinite(velocity_x))
    assert np.all(np.isfinite(velocity_y))
    assert np.min(1.0 + height) > 0.0
    assert np.std(height[-1]) > 1.0e-3
    assert np.std(velocity_x[-1]) > 1.0e-3
    assert np.linalg.norm(height[-1] - height[len(height) // 2]) > 1.0e-1


def test_shallow_water_3d_domain_rejected(tmp_path):
    config = SimulationConfig(
        preset="shallow_water",
        parameters={
            "gravity": 1.0,
            "mean_depth": 1.0,
            "drag": 0.01,
            "viscosity": 0.002,
            "coriolis": 0.0,
        },
        domain=DomainConfig(
            type="box",
            params={"size": [1.0, 1.0, 1.0], "mesh_resolution": [4, 4, 4]},
        ),
        inputs={
            "height": InputConfig(initial_condition=constant(0.0)),
            "velocity": InputConfig(initial_condition=vector_zero()),
        },
        boundary_conditions={
            "height": boundary_field_config({}, periodic_axes=(0, 1, 2)),
            "velocity": boundary_field_config({}, periodic_axes=(0, 1, 2)),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4, 4],
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(height="scalar", velocity="components"),
        ),
        solver=nonlinear_mixed_direct_solver_config(),
        time=TimeConfig(dt=0.02, t_end=0.02),
        seed=42,
        coefficients={"bathymetry": constant(0.0)},
    )

    with pytest.raises(ValueError, match="only supports 2D domains"):
        _run_preset(config)


def test_shallow_water_rejects_mismatched_periodic_fields(tmp_path):
    config = SimulationConfig(
        preset="shallow_water",
        parameters={
            "gravity": 1.0,
            "mean_depth": 1.0,
            "drag": 0.01,
            "viscosity": 0.002,
            "coriolis": 0.0,
        },
        domain=DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [6, 6]},
            periodic_maps={
                "bottom_left": PeriodicMapConfig(
                    slave="x-",
                    master="y-",
                    matrix=[[0.0, 1.0], [1.0, 0.0]],
                    offset=[0.0, 0.0],
                ),
                "top_right": PeriodicMapConfig(
                    slave="x+",
                    master="y+",
                    matrix=[[0.0, 1.0], [1.0, 0.0]],
                    offset=[0.0, 0.0],
                ),
            },
        ),
        inputs={
            "height": InputConfig(initial_condition=constant(0.0)),
            "velocity": InputConfig(initial_condition=vector_zero()),
        },
        boundary_conditions={
            "height": boundary_field_config({}, periodic_axes=(0, 1)),
            "velocity": BoundaryFieldConfig(
                sides={
                    "x-": [BoundaryConditionConfig(type="periodic", pair_with="y-")],
                    "x+": [BoundaryConditionConfig(type="periodic", pair_with="y+")],
                    "y-": [BoundaryConditionConfig(type="periodic", pair_with="x-")],
                    "y+": [BoundaryConditionConfig(type="periodic", pair_with="x+")],
                }
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4],
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(height="scalar", velocity="components"),
        ),
        solver=nonlinear_mixed_direct_solver_config(),
        time=TimeConfig(dt=0.02, t_end=0.02),
        seed=42,
        coefficients={"bathymetry": constant(0.0)},
    )

    preset = get_preset(config.preset)
    problem = preset.build_problem(config)
    with pytest.raises(ValueError, match="identical periodic side pairs"):
        problem.load_domain_geometry()


def test_keller_segel_rejects_nonperiodic_boundaries(tmp_path):
    dirichlet_bc = BoundaryConditionConfig(type="dirichlet", value=constant(1.0))
    config = _make_keller_segel_config(
        tmp_path,
        rho_boundary_conditions={
            "x-": dirichlet_bc,
            "x+": dirichlet_bc,
            "y-": dirichlet_bc,
            "y+": dirichlet_bc,
        },
        c_boundary_conditions={
            "x-": dirichlet_bc,
            "x+": dirichlet_bc,
            "y-": dirichlet_bc,
            "y+": dirichlet_bc,
        },
    )

    preset = get_preset(config.preset)
    problem = preset.build_problem(config)
    with pytest.raises(ValueError, match="unsupported operator 'dirichlet'"):
        problem.load_domain_geometry()


def test_keller_segel_rejects_mismatched_periodic_fields(tmp_path):
    config = SimulationConfig(
        preset="keller_segel",
        parameters={
            "D_rho": 1.0,
            "D_c": 0.5,
            "chi0": 10.0,
            "alpha": 1.0,
            "beta": 1.0,
            "r": 1.0,
        },
        domain=DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [6, 6]},
            periodic_maps={
                "bottom_left": PeriodicMapConfig(
                    slave="x-",
                    master="y-",
                    matrix=[[0.0, 1.0], [1.0, 0.0]],
                    offset=[0.0, 0.0],
                ),
                "top_right": PeriodicMapConfig(
                    slave="x+",
                    master="y+",
                    matrix=[[0.0, 1.0], [1.0, 0.0]],
                    offset=[0.0, 0.0],
                ),
            },
        ),
        inputs={
            "rho": InputConfig(initial_condition=constant(1.0)),
            "c": InputConfig(initial_condition=constant(1.0)),
        },
        boundary_conditions={
            "rho": boundary_field_config({}, periodic_axes=(0, 1)),
            "c": BoundaryFieldConfig(
                sides={
                    "x-": [BoundaryConditionConfig(type="periodic", pair_with="y-")],
                    "x+": [BoundaryConditionConfig(type="periodic", pair_with="y+")],
                    "y-": [BoundaryConditionConfig(type="periodic", pair_with="x-")],
                    "y+": [BoundaryConditionConfig(type="periodic", pair_with="x+")],
                }
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4],
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(rho="scalar", c="scalar"),
        ),
        solver=direct_solver_config(CONSTANT_LHS_SCALAR_SPD),
        time=TimeConfig(dt=0.02, t_end=0.02),
        seed=42,
    )

    preset = get_preset(config.preset)
    problem = preset.build_problem(config)
    with pytest.raises(ValueError, match="identical periodic side pairs"):
        problem.load_domain_geometry()


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

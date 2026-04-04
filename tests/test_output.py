"""Tests for output format writers and FrameWriter coordinator."""

import json
import logging
import shutil
from dataclasses import replace

import numpy as np
import pytest
from dolfinx import fem

from plm_data.core.config import (
    BoundaryConditionConfig,
    DomainConfig,
    FieldExpressionConfig,
    InputConfig,
    OutputConfig,
    SimulationConfig,
    TimeConfig,
)
from plm_data.core.formats.vtk_writer import VTKWriter
from plm_data.core.mesh import create_domain
from plm_data.core.output import FrameWriter
from plm_data.core.solver_strategies import CONSTANT_LHS_SCALAR_SPD
from plm_data.presets import get_preset
from tests.conftest import output_fields, scalar_expr
from tests.preset_matrix import (
    boundary_field_config,
    constant,
    flow_solver_config,
    solver_config,
)

_has_ffmpeg = shutil.which("ffmpeg") is not None


def _scalar_heat_config(tmp_path, formats):
    """Build a minimal scalar heat config with the given output formats."""

    return SimulationConfig(
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
                    "gaussian_bump", sigma=0.1, amplitude=1.0, center=[0.5, 0.5]
                ),
            )
        },
        boundary_conditions={
            "u": boundary_field_config(
                {
                    "x-": _neumann_zero(),
                    "x+": _neumann_zero(),
                    "y-": _neumann_zero(),
                    "y+": _neumann_zero(),
                }
            )
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4],
            num_frames=2,
            formats=formats,
            fields=output_fields(u="scalar"),
        ),
        solver=solver_config(
            CONSTANT_LHS_SCALAR_SPD,
            serial={"ksp_type": "preonly", "pc_type": "lu"},
        ),
        time=TimeConfig(dt=0.01, t_end=0.01),
        seed=42,
        coefficients={"kappa": constant(0.01)},
    )


def _vector_stokes_config(tmp_path, formats):
    """Build a Stokes config (velocity vector + pressure scalar)."""

    return SimulationConfig(
        preset="stokes",
        parameters={"nu": 1.0},
        domain=DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
        ),
        inputs={
            "velocity": InputConfig(
                source=scalar_expr("none"),
            )
        },
        boundary_conditions={
            "velocity": boundary_field_config(
                {
                    "x-": _dirichlet_vector_zero(),
                    "x+": _dirichlet_vector_zero(),
                    "y-": _dirichlet_vector_zero(),
                    "y+": _dirichlet_vector([1.0, 0.0]),
                }
            )
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4],
            num_frames=1,
            formats=formats,
            fields=output_fields(velocity="components", pressure="scalar"),
        ),
        solver=flow_solver_config(),
        seed=42,
    )


def _neumann_zero():
    return BoundaryConditionConfig(
        type="neumann",
        value=FieldExpressionConfig(type="constant", params={"value": 0.0}),
    )


def _dirichlet_vector_zero():
    return BoundaryConditionConfig(
        type="dirichlet",
        value=FieldExpressionConfig(
            components={
                "x": FieldExpressionConfig(type="constant", params={"value": 0.0}),
                "y": FieldExpressionConfig(type="constant", params={"value": 0.0}),
            }
        ),
    )


def _dirichlet_vector(values):
    return BoundaryConditionConfig(
        type="dirichlet",
        value=FieldExpressionConfig(
            components={
                "x": FieldExpressionConfig(
                    type="constant", params={"value": values[0]}
                ),
                "y": FieldExpressionConfig(
                    type="constant", params={"value": values[1]}
                ),
            }
        ),
    )


def _create_scalar_function(config):
    """Create a mesh and a scalar function with f(x) = x[0] for testing."""
    domain_geom = create_domain(config.domain)
    V = fem.functionspace(domain_geom.mesh, ("Lagrange", 1))
    f = fem.Function(V, name="u")
    f.interpolate(lambda x: x[0])
    return f


def _write_scalar_frames(writer, func, num_frames=2):
    """Write multiple frames of a scalar function."""
    for i in range(num_frames):
        writer.write_frame({"u": func}, t=float(i))
    writer.finalize()


class TestNumpyWriter:
    def test_scalar_output(self, tmp_path, caplog):
        config = _scalar_heat_config(tmp_path, formats=["numpy"])
        output_dir = tmp_path / "out"
        spec = get_preset("heat").spec
        writer = FrameWriter(output_dir, config, spec)

        func = _create_scalar_function(config)
        with caplog.at_level(logging.WARNING, logger="plm_data"):
            _write_scalar_frames(writer, func, num_frames=3)

        npy = np.load(output_dir / "u.npy")
        assert npy.shape == (3, 4, 4)
        assert np.max(np.abs(npy)) > 0

        meta = json.loads((output_dir / "frames_meta.json").read_text())
        stagnation = meta["diagnostics"]["stagnation"]
        output_health = meta["diagnostics"]["output_health"]
        assert meta["num_frames"] == 3
        assert meta["times"] == [0.0, 1.0, 2.0]
        assert meta["field_names"] == ["u"]
        assert meta["diagnostics"]["overall_health"]["status"] == "warn"
        assert output_health["status"] == "warn"
        assert stagnation["applied"] is True
        assert stagnation["checked_fields"] == ["u"]
        assert stagnation["skipped_static_fields"] == []
        assert stagnation["stagnant_fields"] == ["u"]
        assert meta["timings"]["frame_count"] == 3
        assert meta["timings"]["grid_interpolation_calls"] == 3
        assert any(
            "Health check: field 'u' appears stagnant" in record.message
            for record in caplog.records
        )

        assert not (output_dir / "u.gif").exists()
        assert not (output_dir / "u.mp4").exists()
        assert not (output_dir / "u.bp").exists()

    def test_static_fields_are_skipped(self, tmp_path, caplog):
        config = _scalar_heat_config(tmp_path, formats=["numpy"])
        output_dir = tmp_path / "out"
        spec = replace(get_preset("heat").spec, static_fields=["u"])
        writer = FrameWriter(output_dir, config, spec)

        func = _create_scalar_function(config)
        with caplog.at_level(logging.WARNING, logger="plm_data"):
            _write_scalar_frames(writer, func, num_frames=3)

        meta = json.loads((output_dir / "frames_meta.json").read_text())
        stagnation = meta["diagnostics"]["stagnation"]
        output_health = meta["diagnostics"]["output_health"]
        assert meta["diagnostics"]["overall_health"]["status"] == "pass"
        assert output_health["status"] == "pass"
        assert stagnation["applied"] is True
        assert stagnation["reason"] is None
        assert stagnation["checked_fields"] == []
        assert stagnation["skipped_static_fields"] == ["u"]
        assert stagnation["stagnant_fields"] == []
        assert not any("Health check:" in record.message for record in caplog.records)

    def test_nonfinite_base_field_fails_fast(self, tmp_path):
        config = _scalar_heat_config(tmp_path, formats=["numpy"])
        output_dir = tmp_path / "out"
        spec = get_preset("heat").spec
        writer = FrameWriter(output_dir, config, spec)

        func = _create_scalar_function(config)
        func.x.array[0] = np.nan

        with pytest.raises(ValueError, match="Non-finite values detected"):
            writer.write_frame({"u": func}, t=0.0)

    def test_vector_components_output(self, tmp_path):
        config = _vector_stokes_config(tmp_path, formats=["numpy"])
        output_dir = tmp_path / "out"
        spec = get_preset("stokes").spec

        preset = get_preset("stokes")
        problem = preset.build_problem(config)
        writer = FrameWriter(output_dir, config, spec)
        problem.run(writer)
        writer.finalize()

        vx = np.load(output_dir / "velocity_x.npy")
        vy = np.load(output_dir / "velocity_y.npy")
        p = np.load(output_dir / "pressure.npy")
        assert vx.shape == (1, 4, 4)
        assert vy.shape == (1, 4, 4)
        assert p.shape == (1, 4, 4)
        assert np.max(np.abs(vx)) > 0

        meta = json.loads((output_dir / "frames_meta.json").read_text())
        assert meta["timings"]["grid_interpolation_calls"] == 2


class TestGifWriter:
    @pytest.fixture(autouse=True)
    def _require_matplotlib(self):
        pytest.importorskip("matplotlib")

    def test_scalar_output(self, tmp_path):
        config = _scalar_heat_config(tmp_path, formats=["gif"])
        output_dir = tmp_path / "out"
        spec = get_preset("heat").spec
        writer = FrameWriter(output_dir, config, spec)

        func = _create_scalar_function(config)
        _write_scalar_frames(writer, func, num_frames=2)

        gif_path = output_dir / "u.gif"
        assert gif_path.exists()
        assert gif_path.stat().st_size > 0

        with open(gif_path, "rb") as f:
            magic = f.read(6)
        assert magic in (b"GIF87a", b"GIF89a")
        assert not (output_dir / "u.npy").exists()

    def test_vector_components_output(self, tmp_path):
        config = _vector_stokes_config(tmp_path, formats=["gif"])
        output_dir = tmp_path / "out"
        spec = get_preset("stokes").spec

        preset = get_preset("stokes")
        problem = preset.build_problem(config)
        writer = FrameWriter(output_dir, config, spec)
        problem.run(writer)
        writer.finalize()

        for filename in ["velocity_x.gif", "velocity_y.gif", "pressure.gif"]:
            path = output_dir / filename
            assert path.exists()
            assert path.stat().st_size > 0


class TestVTKWriter:
    def test_scalar_output_uses_native_fem_mesh(self, tmp_path):
        config = _scalar_heat_config(tmp_path, formats=["vtk"])
        output_dir = tmp_path / "out"
        spec = get_preset("heat").spec
        writer = FrameWriter(output_dir, config, spec)

        func = _create_scalar_function(config)
        _write_scalar_frames(writer, func, num_frames=2)

        paraview_dir = output_dir / "paraview"
        assert (paraview_dir / "u.pvd").exists()
        assert not (paraview_dir / "solution.pvd").exists()
        assert any(path.suffix in {".vtu", ".pvtu"} for path in paraview_dir.iterdir())
        assert not (output_dir / "u.npy").exists()

        meta = json.loads((output_dir / "frames_meta.json").read_text())
        assert meta["timings"]["vtk_write_seconds"] > 0.0

    def test_noncg_fields_are_visualized_on_same_mesh(self, tmp_path):
        domain = create_domain(
            DomainConfig(
                type="rectangle",
                params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
            )
        )
        V = fem.functionspace(domain.mesh, ("N1curl", 1))
        field = fem.Function(V, name="electric_field")
        field.interpolate(
            lambda x: np.vstack((np.zeros_like(x[0]), np.ones_like(x[1])))
        )

        writer = VTKWriter(tmp_path)
        writer.on_frame({"electric_field": field}, t=0.0)
        writer.finalize()

        paraview_dir = tmp_path / "paraview"
        assert (paraview_dir / "electric_field.pvd").exists()
        assert any(path.suffix in {".vtu", ".pvtu"} for path in paraview_dir.iterdir())

    def test_mixed_element_outputs_use_separate_series(self, tmp_path):
        config = _vector_stokes_config(tmp_path, formats=["vtk"])
        output_dir = tmp_path / "out"
        spec = get_preset("stokes").spec

        preset = get_preset("stokes")
        problem = preset.build_problem(config)
        writer = FrameWriter(output_dir, config, spec)
        problem.run(writer)
        writer.finalize()

        paraview_dir = output_dir / "paraview"
        assert (paraview_dir / "velocity.pvd").exists()
        assert (paraview_dir / "pressure.pvd").exists()
        assert not (paraview_dir / "solution.pvd").exists()
        assert any(path.suffix in {".vtu", ".pvtu"} for path in paraview_dir.iterdir())


@pytest.mark.skipif(not _has_ffmpeg, reason="ffmpeg not installed")
class TestVideoWriter:
    @pytest.fixture(autouse=True)
    def _require_matplotlib(self):
        pytest.importorskip("matplotlib")

    def test_scalar_output(self, tmp_path):
        config = _scalar_heat_config(tmp_path, formats=["video"])
        output_dir = tmp_path / "out"
        spec = get_preset("heat").spec
        writer = FrameWriter(output_dir, config, spec)

        func = _create_scalar_function(config)
        _write_scalar_frames(writer, func, num_frames=2)

        video_path = output_dir / "u.mp4"
        assert video_path.exists()
        assert video_path.stat().st_size > 0

    def test_vector_components_output(self, tmp_path):
        config = _vector_stokes_config(tmp_path, formats=["video"])
        output_dir = tmp_path / "out"
        spec = get_preset("stokes").spec

        preset = get_preset("stokes")
        problem = preset.build_problem(config)
        writer = FrameWriter(output_dir, config, spec)
        problem.run(writer)
        writer.finalize()

        for filename in ["velocity_x.mp4", "velocity_y.mp4", "pressure.mp4"]:
            path = output_dir / filename
            assert path.exists()
            assert path.stat().st_size > 0

"""Tests for output format writers and FrameWriter coordinator."""

import json
import shutil

import numpy as np
import pytest
from dolfinx import fem

from plm_data.core.config import (
    DomainConfig,
    FieldConfig,
    FieldOutputConfig,
    OutputConfig,
    SimulationConfig,
    SolverConfig,
    TimeConfig,
)
from plm_data.core.mesh import create_domain
from plm_data.core.output import FrameWriter
from plm_data.presets import get_preset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_has_ffmpeg = shutil.which("ffmpeg") is not None


def _scalar_heat_config(tmp_path, formats):
    """Build a minimal scalar heat config with the given output formats."""
    from tests.conftest import scalar_expr

    return SimulationConfig(
        preset="heat",
        parameters={"kappa": 0.01},
        domain=DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
        ),
        fields={
            "u": FieldConfig(
                boundary_conditions={
                    "x-": _neumann_zero(),
                    "x+": _neumann_zero(),
                    "y-": _neumann_zero(),
                    "y+": _neumann_zero(),
                },
                source=scalar_expr("none"),
                initial_condition=scalar_expr(
                    "gaussian_bump", sigma=0.1, amplitude=1.0, center=[0.5, 0.5]
                ),
                output=FieldOutputConfig(mode="scalar"),
            )
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4],
            num_frames=2,
            formats=formats,
        ),
        solver=SolverConfig(options={"ksp_type": "preonly", "pc_type": "lu"}),
        time=TimeConfig(dt=0.01, t_end=0.01),
        seed=42,
    )


def _vector_stokes_config(tmp_path, formats):
    """Build a Stokes config (velocity vector + pressure scalar)."""
    from tests.conftest import scalar_expr

    return SimulationConfig(
        preset="stokes",
        parameters={"nu": 1.0},
        domain=DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
        ),
        fields={
            "velocity": FieldConfig(
                boundary_conditions={
                    "x-": _dirichlet_vector_zero(),
                    "x+": _dirichlet_vector_zero(),
                    "y-": _dirichlet_vector_zero(),
                    "y+": _dirichlet_vector([1.0, 0.0]),
                },
                source=scalar_expr("none"),
                output=FieldOutputConfig(mode="components"),
            ),
            "pressure": FieldConfig(output=FieldOutputConfig(mode="scalar")),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4],
            num_frames=1,
            formats=formats,
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


def _neumann_zero():
    from plm_data.core.config import BoundaryConditionConfig, FieldExpressionConfig

    return BoundaryConditionConfig(
        type="neumann",
        value=FieldExpressionConfig(type="constant", params={"value": 0.0}),
    )


def _dirichlet_vector_zero():
    from plm_data.core.config import BoundaryConditionConfig, FieldExpressionConfig

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
    from plm_data.core.config import BoundaryConditionConfig, FieldExpressionConfig

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


# ---------------------------------------------------------------------------
# Numpy format tests
# ---------------------------------------------------------------------------


class TestNumpyWriter:
    def test_scalar_output(self, tmp_path):
        config = _scalar_heat_config(tmp_path, formats=["numpy"])
        output_dir = tmp_path / "out"
        spec = get_preset("heat").spec
        writer = FrameWriter(output_dir, config, spec)

        func = _create_scalar_function(config)
        _write_scalar_frames(writer, func, num_frames=2)

        # .npy file exists with correct shape
        npy = np.load(output_dir / "u.npy")
        assert npy.shape == (2, 4, 4)
        assert np.max(np.abs(npy)) > 0

        # Metadata
        meta = json.loads((output_dir / "frames_meta.json").read_text())
        assert meta["num_frames"] == 2
        assert meta["times"] == [0.0, 1.0]
        assert meta["field_names"] == ["u"]

        # No other format files
        assert not (output_dir / "u.gif").exists()
        assert not (output_dir / "u.mp4").exists()
        assert not (output_dir / "u.bp").exists()

    def test_vector_components_output(self, tmp_path):
        config = _vector_stokes_config(tmp_path, formats=["numpy"])
        output_dir = tmp_path / "out"
        spec = get_preset("stokes").spec

        # Run the actual preset to get proper mixed-space functions
        preset = get_preset("stokes")
        problem = preset.build_problem(config)
        writer = FrameWriter(output_dir, config, spec)
        problem.run(writer)
        writer.finalize()

        # Component arrays exist
        vx = np.load(output_dir / "velocity_x.npy")
        vy = np.load(output_dir / "velocity_y.npy")
        p = np.load(output_dir / "pressure.npy")
        assert vx.shape == (1, 4, 4)
        assert vy.shape == (1, 4, 4)
        assert p.shape == (1, 4, 4)

        # At least velocity_x should be non-zero (lid-driven cavity)
        assert np.max(np.abs(vx)) > 0


# ---------------------------------------------------------------------------
# GIF format tests
# ---------------------------------------------------------------------------


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

        # Check GIF magic bytes
        with open(gif_path, "rb") as f:
            magic = f.read(6)
        assert magic in (b"GIF87a", b"GIF89a")

        # No numpy output
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

        assert (output_dir / "velocity_x.gif").exists()
        assert (output_dir / "velocity_y.gif").exists()
        assert (output_dir / "pressure.gif").exists()


# ---------------------------------------------------------------------------
# Video format tests
# ---------------------------------------------------------------------------


class TestVideoWriter:
    @pytest.fixture(autouse=True)
    def _require_deps(self):
        pytest.importorskip("matplotlib")
        if not _has_ffmpeg:
            pytest.skip("ffmpeg not available")

    def test_scalar_output(self, tmp_path):
        config = _scalar_heat_config(tmp_path, formats=["video"])
        output_dir = tmp_path / "out"
        spec = get_preset("heat").spec
        writer = FrameWriter(output_dir, config, spec)

        func = _create_scalar_function(config)
        _write_scalar_frames(writer, func, num_frames=2)

        mp4_path = output_dir / "u.mp4"
        assert mp4_path.exists()
        assert mp4_path.stat().st_size > 0

        # No other formats
        assert not (output_dir / "u.npy").exists()
        assert not (output_dir / "u.gif").exists()


# ---------------------------------------------------------------------------
# VTK format tests
# ---------------------------------------------------------------------------


class TestVTKWriter:
    @pytest.fixture(autouse=True)
    def _require_pyvista(self):
        pytest.importorskip("pyvista")

    def test_scalar_output(self, tmp_path):
        config = _scalar_heat_config(tmp_path, formats=["vtk"])
        output_dir = tmp_path / "out"
        pv_dir = output_dir / "paraview"
        spec = get_preset("heat").spec
        writer = FrameWriter(output_dir, config, spec)

        func = _create_scalar_function(config)
        _write_scalar_frames(writer, func, num_frames=2)

        # Everything lives inside paraview/
        assert pv_dir.is_dir()
        assert (pv_dir / "u.pvd").exists()
        assert len(list(pv_dir.glob("u_*.vtu"))) == 2

        # No numpy files
        assert not (output_dir / "u.npy").exists()

        # Metadata still written
        assert (output_dir / "frames_meta.json").exists()

    def test_vector_output(self, tmp_path):
        config = _vector_stokes_config(tmp_path, formats=["vtk"])
        output_dir = tmp_path / "out"
        pv_dir = output_dir / "paraview"
        spec = get_preset("stokes").spec

        preset = get_preset("stokes")
        problem = preset.build_problem(config)
        writer = FrameWriter(output_dir, config, spec)
        problem.run(writer)
        writer.finalize()

        # PVD files for base fields (not component-split)
        assert (pv_dir / "velocity.pvd").exists()
        assert (pv_dir / "pressure.pvd").exists()
        assert not (pv_dir / "velocity_x.pvd").exists()
        assert not (pv_dir / "velocity_y.pvd").exists()

        # .vtu frame files exist
        assert len(list(pv_dir.glob("velocity_*.vtu"))) >= 1
        assert len(list(pv_dir.glob("pressure_*.vtu"))) >= 1

    def test_skips_interpolation(self, tmp_path):
        config = _scalar_heat_config(tmp_path, formats=["vtk"])
        output_dir = tmp_path / "out"
        spec = get_preset("heat").spec
        writer = FrameWriter(output_dir, config, spec)

        func = _create_scalar_function(config)
        _write_scalar_frames(writer, func, num_frames=1)

        # Interpolation cache was never created
        assert writer._interp_cache is None


# ---------------------------------------------------------------------------
# Multi-format tests
# ---------------------------------------------------------------------------


class TestMultiFormat:
    @pytest.fixture(autouse=True)
    def _require_deps(self):
        pytest.importorskip("matplotlib")
        pytest.importorskip("pyvista")

    def test_numpy_gif_vtk(self, tmp_path):
        config = _scalar_heat_config(tmp_path, formats=["numpy", "gif", "vtk"])
        output_dir = tmp_path / "out"
        spec = get_preset("heat").spec
        writer = FrameWriter(output_dir, config, spec)

        func = _create_scalar_function(config)
        _write_scalar_frames(writer, func, num_frames=2)

        # All three formats present
        npy = np.load(output_dir / "u.npy")
        assert npy.shape == (2, 4, 4)
        assert (output_dir / "u.gif").exists()
        pv_dir = output_dir / "paraview"
        assert (pv_dir / "u.pvd").exists()
        assert len(list(pv_dir.glob("u_*.vtu"))) == 2

        # Metadata correct
        meta = json.loads((output_dir / "frames_meta.json").read_text())
        assert meta["num_frames"] == 2


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestConfigFormatValidation:
    def test_empty_formats_rejected(self, tmp_path):
        import yaml

        data = _base_yaml_dict()
        data["output"]["formats"] = []
        p = tmp_path / "empty_formats.yaml"
        p.write_text(yaml.dump(data))

        from plm_data.core.config import load_config

        with pytest.raises(ValueError, match="at least one format"):
            load_config(p)

    def test_unknown_format_rejected(self, tmp_path):
        import yaml

        data = _base_yaml_dict()
        data["output"]["formats"] = ["numpy", "hdf5"]
        p = tmp_path / "bad_format.yaml"
        p.write_text(yaml.dump(data))

        from plm_data.core.config import load_config

        with pytest.raises(ValueError, match="Unknown output format"):
            load_config(p)

    def test_duplicate_format_rejected(self, tmp_path):
        import yaml

        data = _base_yaml_dict()
        data["output"]["formats"] = ["numpy", "numpy"]
        p = tmp_path / "dup_format.yaml"
        p.write_text(yaml.dump(data))

        from plm_data.core.config import load_config

        with pytest.raises(ValueError, match="duplicates"):
            load_config(p)

    def test_valid_multi_format_accepted(self, tmp_path):
        import yaml

        data = _base_yaml_dict()
        data["output"]["formats"] = ["numpy", "gif", "vtk"]
        p = tmp_path / "multi_format.yaml"
        p.write_text(yaml.dump(data))

        from plm_data.core.config import load_config

        cfg = load_config(p)
        assert cfg.output.formats == ["numpy", "gif", "vtk"]


class TestNeedsGridInterpolation:
    def test_numpy_needs_interpolation(self):
        cfg = OutputConfig(
            path="/tmp", resolution=[4, 4], num_frames=1, formats=["numpy"]
        )
        assert cfg.needs_grid_interpolation is True

    def test_vtk_skips_interpolation(self):
        cfg = OutputConfig(
            path="/tmp", resolution=[4, 4], num_frames=1, formats=["vtk"]
        )
        assert cfg.needs_grid_interpolation is False

    def test_vtk_gif_needs_interpolation(self):
        cfg = OutputConfig(
            path="/tmp", resolution=[4, 4], num_frames=1, formats=["vtk", "gif"]
        )
        assert cfg.needs_grid_interpolation is True


# ---------------------------------------------------------------------------
# Shared YAML dict for config validation tests
# ---------------------------------------------------------------------------


def _base_yaml_dict():
    return {
        "preset": "poisson",
        "parameters": {"kappa": 1.0, "f_amplitude": 1.0},
        "domain": {
            "type": "rectangle",
            "size": [1.0, 1.0],
            "mesh_resolution": [4, 4],
        },
        "fields": {
            "u": {
                "boundary_conditions": {
                    "x-": {"type": "dirichlet", "value": 0.0},
                    "x+": {"type": "dirichlet", "value": 0.0},
                    "y-": {"type": "dirichlet", "value": 0.0},
                    "y+": {"type": "dirichlet", "value": 0.0},
                },
                "source": {"type": "none", "params": {}},
                "output": "scalar",
            },
        },
        "output": {
            "path": "./output",
            "resolution": [8, 8],
            "num_frames": 1,
            "formats": ["numpy"],
        },
        "solver": {
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    }

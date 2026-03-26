"""Tests for plm_data.core.output."""

import numpy as np
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


def test_frame_writer(tmp_path, heat_config):
    domain_geom = create_domain(heat_config.domain)
    V = fem.functionspace(domain_geom.mesh, ("Lagrange", 1))
    f = fem.Function(V)
    f.x.array[:] = 1.0  # type: ignore[reportAttributeAccessIssue]

    output_dir = tmp_path / "test_output"
    heat_config.output.path = tmp_path
    writer = FrameWriter(output_dir, heat_config, get_preset("heat").spec)
    writer.write_frame({"u": f}, t=0.0)  # type: ignore[reportArgumentType]
    writer.finalize()

    npy_path = output_dir / "u.npy"
    assert npy_path.exists()
    arr = np.load(npy_path)
    assert arr.shape == (1, *heat_config.output.resolution)

    meta_path = output_dir / "frames_meta.json"
    assert meta_path.exists()


def test_frame_writer_hcurl_vector_field(tmp_path):
    config = SimulationConfig(
        preset="maxwell_pulse",
        parameters={},
        domain=DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [6, 6]},
        ),
        fields={
            "electric_field": FieldConfig(output=FieldOutputConfig(mode="components"))
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[6, 6],
            num_frames=1,
            formats=["numpy"],
        ),
        solver=SolverConfig(options={}),
        time=TimeConfig(dt=0.01, t_end=0.01),
    )

    domain_geom = create_domain(config.domain)
    V = fem.functionspace(domain_geom.mesh, ("N1curl", 1))
    E = fem.Function(V, name="electric_field")
    E.interpolate(lambda x: np.vstack((x[1], -x[0])))

    output_dir = tmp_path / "test_hcurl_output"
    writer = FrameWriter(output_dir, config, get_preset("maxwell_pulse").spec)
    writer.write_frame({"electric_field": E}, t=0.0)
    writer.finalize()

    ex = np.load(output_dir / "electric_field_x.npy")
    ey = np.load(output_dir / "electric_field_y.npy")
    assert ex.shape == (1, *config.output.resolution)
    assert ey.shape == (1, *config.output.resolution)
    assert np.max(np.abs(ex)) > 0
    assert np.max(np.abs(ey)) > 0

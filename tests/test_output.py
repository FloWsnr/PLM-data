"""Tests for plm_data.core.output."""

import numpy as np
from dolfinx import fem

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

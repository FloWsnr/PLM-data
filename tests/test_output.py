"""Tests for plm_data.core.output."""

import numpy as np
from dolfinx import fem

from plm_data.core.mesh import create_mesh
from plm_data.core.output import FrameWriter


def test_frame_writer(tmp_path, heat_config):
    msh = create_mesh(heat_config.domain)
    V = fem.functionspace(msh, ("Lagrange", 1))
    f = fem.Function(V)
    f.x.array[:] = 1.0

    output_dir = tmp_path / "test_output"
    heat_config.output.path = tmp_path
    writer = FrameWriter(output_dir, heat_config)
    writer.write_frame({"u": f}, t=0.0)
    writer.finalize()

    npy_path = output_dir / "frames" / "u" / "000000.npy"
    assert npy_path.exists()
    arr = np.load(npy_path)
    assert arr.shape == tuple(heat_config.output_resolution)

    meta_path = output_dir / "frames_meta.json"
    assert meta_path.exists()

"""VTK/Paraview output format using pyvista."""

import importlib
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from dolfinx import fem, plot

from plm_data.core.logging import get_logger


def _func_to_point_data(func: fem.Function) -> np.ndarray:
    """Extract point data from a DOLFINx function for pyvista."""
    values = func.x.array.real.copy()
    value_shape = func.function_space.element.value_shape
    if len(value_shape) == 0 or value_shape == (1,):
        # Scalar field
        return values
    # Vector field — pad to 3D for pyvista
    gdim = value_shape[0]
    values = values.reshape(-1, gdim)
    if gdim < 3:
        padded = np.zeros((values.shape[0], 3), dtype=values.dtype)
        padded[:, :gdim] = values
        return padded
    return values


class VTKWriter:
    """Writes FEM fields to .vtu files with a .pvd collection for Paraview."""

    def __init__(self, output_dir: Path):
        importlib.import_module("pyvista")

        self._paraview_dir = output_dir / "paraview"
        self._logger = get_logger("output.vtk")
        self._frame_idx = 0
        self._pvd_entries: dict[str, list[tuple[float, str]]] = {}

    def on_frame(self, fields: dict[str, fem.Function], t: float) -> None:
        """Write one frame of base fields to .vtu files."""
        pyvista = importlib.import_module("pyvista")

        self._paraview_dir.mkdir(parents=True, exist_ok=True)

        for name, func in fields.items():
            topology, cell_types, geometry = plot.vtk_mesh(func.function_space)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
            grid.point_data[name] = _func_to_point_data(func)

            vtu_name = f"{name}_{self._frame_idx:04d}.vtu"
            grid.save(str(self._paraview_dir / vtu_name))
            self._pvd_entries.setdefault(name, []).append((t, vtu_name))

        self._frame_idx += 1

    def finalize(self) -> None:
        """Write .pvd collection files for each field."""
        for name, entries in self._pvd_entries.items():
            self._write_pvd(name, entries)
        self._logger.info(
            "  VTK output: %d field(s), %d frame(s) → %s",
            len(self._pvd_entries),
            self._frame_idx,
            self._paraview_dir,
        )

    def _write_pvd(self, name: str, entries: list[tuple[float, str]]) -> None:
        """Write a PVD collection XML file."""
        root = ET.Element(
            "VTKFile", type="Collection", version="0.1", byte_order="LittleEndian"
        )
        collection = ET.SubElement(root, "Collection")
        for t, vtu_name in entries:
            ET.SubElement(
                collection,
                "DataSet",
                timestep=str(t),
                file=vtu_name,
            )
        tree = ET.ElementTree(root)
        ET.indent(tree)
        tree.write(self._paraview_dir / f"{name}.pvd", xml_declaration=True)

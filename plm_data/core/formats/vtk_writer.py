"""VTK/Paraview output using DOLFINx's native mesh/function writer."""

from pathlib import Path

from dolfinx import fem
from dolfinx.io import VTKFile
from dolfinx.mesh import Mesh

from plm_data.core.logging import get_logger


class VTKWriter:
    """Write native FEM mesh/function output for ParaView."""

    def __init__(self, output_dir: Path):
        self._paraview_dir = output_dir / "paraview"
        self._logger = get_logger("output.vtk")
        self._writers: dict[str, VTKFile] = {}
        self._meshes: dict[str, Mesh] = {}
        self._frame_idx = 0
        self._mesh_written: set[str] = set()
        self._vis_cache: dict[str, fem.Function] = {}

    def _is_vtk_native_function(self, func: fem.Function) -> bool:
        """Return whether DOLFINx can write the function directly to VTK."""
        basix_element = getattr(func.function_space.element, "basix_element", None)
        family = getattr(getattr(basix_element, "family", None), "name", None)
        return family == "P"

    def _visualization_function(
        self, output_name: str, func: fem.Function
    ) -> fem.Function:
        """Return a VTK-compatible function on the same mesh."""
        mesh = self._ensure_output_mesh(output_name, func.function_space.mesh)

        if self._is_vtk_native_function(func):
            return func

        if output_name not in self._vis_cache:
            element = func.function_space.element
            basix_element = getattr(element, "basix_element", None)
            degree = max(1, getattr(basix_element, "degree", 1))
            value_shape = tuple(int(dim) for dim in element.value_shape)

            if value_shape:
                vis_space = fem.functionspace(
                    mesh,
                    ("Discontinuous Lagrange", degree, value_shape),
                )
            else:
                vis_space = fem.functionspace(
                    mesh,
                    ("Discontinuous Lagrange", degree),
                )
            self._vis_cache[output_name] = fem.Function(vis_space, name=output_name)

        vis_func = self._vis_cache[output_name]
        vis_func.interpolate(func)
        return vis_func

    def _ensure_output_mesh(self, output_name: str, mesh: Mesh) -> Mesh:
        """Ensure a VTK series stays attached to one mesh for the whole run."""
        existing_mesh = self._meshes.get(output_name)
        if existing_mesh is None:
            self._meshes[output_name] = mesh
            return mesh
        if existing_mesh is not mesh:
            raise ValueError(f"VTK output '{output_name}' changed mesh during a run.")
        return existing_mesh

    def _writer_for_output(self, output_name: str, mesh: Mesh) -> VTKFile:
        """Create or reuse the ParaView time series for one output field."""
        writer = self._writers.get(output_name)
        if writer is None:
            writer = VTKFile(mesh.comm, self._paraview_dir / f"{output_name}.pvd", "w")
            self._writers[output_name] = writer
        return writer

    def on_frame(self, fields: dict[str, fem.Function], t: float) -> None:
        """Write one frame of FEM fields on the native DOLFINx mesh."""
        if not fields:
            return

        self._paraview_dir.mkdir(parents=True, exist_ok=True)

        for output_name, func in fields.items():
            vis_func = self._visualization_function(output_name, func)
            mesh = self._ensure_output_mesh(output_name, vis_func.function_space.mesh)
            writer = self._writer_for_output(output_name, mesh)

            original_name = vis_func.name
            try:
                vis_func.name = output_name
                if output_name not in self._mesh_written:
                    writer.write_mesh(mesh)
                    self._mesh_written.add(output_name)
                writer.write_function(vis_func, t)
            finally:
                vis_func.name = original_name

        self._frame_idx += 1

    def finalize(self) -> None:
        """Close the VTK writer and log the output location."""
        for writer in self._writers.values():
            writer.close()

        self._logger.info(
            "  VTK output: %d frame(s) → %s",
            self._frame_idx,
            self._paraview_dir,
        )

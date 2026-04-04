"""VTK/Paraview output using DOLFINx's native mesh/function writer."""

from pathlib import Path

from dolfinx import fem
from dolfinx.io import VTKFile

from plm_data.core.logging import get_logger


class VTKWriter:
    """Write native FEM mesh/function output for ParaView."""

    def __init__(self, output_dir: Path):
        self._paraview_dir = output_dir / "paraview"
        self._logger = get_logger("output.vtk")
        self._writer: VTKFile | None = None
        self._frame_idx = 0
        self._mesh_written = False
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
        if self._is_vtk_native_function(func):
            return func

        if output_name not in self._vis_cache:
            element = func.function_space.element
            basix_element = getattr(element, "basix_element", None)
            degree = max(1, getattr(basix_element, "degree", 1))
            value_shape = tuple(int(dim) for dim in element.value_shape)

            if value_shape:
                vis_space = fem.functionspace(
                    func.function_space.mesh,
                    ("Discontinuous Lagrange", degree, value_shape),
                )
            else:
                vis_space = fem.functionspace(
                    func.function_space.mesh,
                    ("Discontinuous Lagrange", degree),
                )
            self._vis_cache[output_name] = fem.Function(vis_space, name=output_name)

        vis_func = self._vis_cache[output_name]
        vis_func.interpolate(func)
        return vis_func

    def on_frame(self, fields: dict[str, fem.Function], t: float) -> None:
        """Write one frame of FEM fields on the native DOLFINx mesh."""
        if not fields:
            return

        self._paraview_dir.mkdir(parents=True, exist_ok=True)

        vtk_functions = [
            self._visualization_function(output_name, func)
            for output_name, func in fields.items()
        ]
        mesh = vtk_functions[0].function_space.mesh
        if self._writer is None:
            self._writer = VTKFile(mesh.comm, self._paraview_dir / "solution.pvd", "w")

        for func in vtk_functions[1:]:
            if func.function_space.mesh is not mesh:
                raise ValueError("VTK output requires all functions to share one mesh.")

        original_names = [func.name for func in vtk_functions]
        try:
            for output_name, func in zip(fields, vtk_functions):
                func.name = output_name
            if not self._mesh_written:
                self._writer.write_mesh(mesh)
                self._mesh_written = True
            self._writer.write_function(vtk_functions, t)
        finally:
            for func, original_name in zip(vtk_functions, original_names):
                func.name = original_name

        self._frame_idx += 1

    def finalize(self) -> None:
        """Close the VTK writer and log the output location."""
        if self._writer is not None:
            self._writer.close()

        self._logger.info(
            "  VTK output: %d frame(s) → %s",
            self._frame_idx,
            self._paraview_dir,
        )

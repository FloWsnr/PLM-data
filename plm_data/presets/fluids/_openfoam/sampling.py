from pathlib import Path
import shutil

import numpy as np
import pyvista as pv

from plm_data.core.spatial_fields import component_labels_for_dim


def _iter_named_blocks(
    data: pv.DataSet | pv.MultiBlock,
    *,
    prefix: str = "",
):
    if isinstance(data, pv.MultiBlock):
        for index in range(data.n_blocks):
            block_name = data.get_block_name(index) or f"block_{index}"
            next_prefix = block_name if not prefix else f"{prefix}/{block_name}"
            block = data[index]
            if block is None:
                continue
            yield from _iter_named_blocks(block, prefix=next_prefix)
        return
    yield prefix, data


def _extract_internal_mesh(data: pv.DataSet | pv.MultiBlock) -> pv.DataSet:
    first_candidate: pv.DataSet | None = None
    for name, block in _iter_named_blocks(data):
        if getattr(block, "n_cells", 0) == 0:
            continue
        if first_candidate is None:
            first_candidate = block
        if name.endswith("internalMesh"):
            return block
    if first_candidate is not None:
        return first_candidate
    raise RuntimeError("OpenFOAM reader did not expose an internal mesh.")


def _openfoam_reader(case_file: Path) -> pv.OpenFOAMReader:
    reader = pv.OpenFOAMReader(str(case_file))
    reader.skip_zero_time = False
    reader.enable_all_patch_arrays()
    return reader


def _read_case_mesh(case_file: Path, *, time_value: float | None = None):
    reader = _openfoam_reader(case_file)
    if time_value is not None:
        reader.set_active_time_value(float(time_value))
    return reader.read()


def _uniform_points_from_bounds(
    *,
    bounds: tuple[float, float, float, float, float, float],
    resolution: tuple[int, ...],
    gdim: int,
) -> tuple[np.ndarray, np.ndarray]:
    if gdim == 2:
        x_values = np.linspace(bounds[0], bounds[1], resolution[0])
        y_values = np.linspace(bounds[2], bounds[3], resolution[1])
        grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="ij")
        z_value = 0.5 * (bounds[4] + bounds[5])
        points = np.column_stack(
            (
                grid_x.ravel(order="C"),
                grid_y.ravel(order="C"),
                np.full(grid_x.size, z_value),
            )
        )
        return points, np.stack((grid_x, grid_y))

    x_values = np.linspace(bounds[0], bounds[1], resolution[0])
    y_values = np.linspace(bounds[2], bounds[3], resolution[1])
    z_values = np.linspace(bounds[4], bounds[5], resolution[2])
    grid_x, grid_y, grid_z = np.meshgrid(
        x_values,
        y_values,
        z_values,
        indexing="ij",
    )
    points = np.column_stack(
        (
            grid_x.ravel(order="C"),
            grid_y.ravel(order="C"),
            grid_z.ravel(order="C"),
        )
    )
    return points, np.stack((grid_x, grid_y, grid_z))


def _sample_openfoam_field_data(
    *,
    internal_mesh: pv.DataSet,
    sample_points: np.ndarray,
) -> tuple[pv.DataSet, np.ndarray]:
    point_cloud = pv.PolyData(sample_points)
    sampled = point_cloud.sample(
        internal_mesh,
        pass_cell_data=True,
        pass_point_data=True,
    )
    if "vtkValidPointMask" in sampled.point_data:
        valid_mask = np.asarray(sampled.point_data["vtkValidPointMask"], dtype=bool)
    else:
        valid_mask = np.ones(sample_points.shape[0], dtype=bool)
    return sampled, valid_mask


def _sampled_scalar_array(
    sampled: pv.DataSet,
    *,
    name: str,
    resolution: tuple[int, ...],
    valid_mask: np.ndarray,
    normalize_mean: bool = False,
    subtract_offset: float = 0.0,
) -> np.ndarray:
    values = np.asarray(sampled.point_data[name], dtype=float).copy()
    values[~valid_mask] = 0.0
    if subtract_offset != 0.0 and np.any(valid_mask):
        values[valid_mask] = values[valid_mask] - subtract_offset
    if normalize_mean and np.any(valid_mask):
        values[valid_mask] = values[valid_mask] - float(np.mean(values[valid_mask]))
    return values.reshape(resolution, order="C")


def _sampled_density_array_from_pressure_temperature(
    sampled: pv.DataSet,
    *,
    resolution: tuple[int, ...],
    valid_mask: np.ndarray,
    gas_constant: float,
) -> np.ndarray:
    pressure = np.asarray(sampled.point_data["p"], dtype=float)
    temperature = np.asarray(sampled.point_data["T"], dtype=float)
    density = np.zeros_like(pressure)
    safe_mask = valid_mask & (np.abs(temperature) > 0.0)
    density[safe_mask] = pressure[safe_mask] / (gas_constant * temperature[safe_mask])
    return density.reshape(resolution, order="C")


def _sampled_vector_components(
    sampled: pv.DataSet,
    *,
    name: str,
    resolution: tuple[int, ...],
    valid_mask: np.ndarray,
    gdim: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(sampled.point_data[name], dtype=float).copy()
    if values.ndim != 2 or values.shape[1] < gdim:
        raise ValueError(
            f"OpenFOAM field '{name}' produced shape {values.shape}; expected at "
            f"least {gdim} vector components."
        )
    values[~valid_mask, :] = 0.0
    arrays: dict[str, np.ndarray] = {}
    for axis, label in enumerate(component_labels_for_dim(gdim)):
        arrays[label] = values[:, axis].reshape(resolution, order="C")
    return arrays


def _remove_processor_dirs(case_dir: Path) -> None:
    for path in case_dir.iterdir():
        if path.is_dir() and path.name.startswith("processor"):
            shutil.rmtree(path)

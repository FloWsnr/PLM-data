"""Shared helpers for preset runtime coverage tests."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import importlib.util

import numpy as np
from numpy.typing import NDArray

from plm_data.core.config import (
    BoundaryConditionConfig,
    BoundaryFieldConfig,
    DomainConfig,
    FieldExpressionConfig,
    InputConfig,
    OutputConfig,
    OutputSelectionConfig,
    SimulationConfig,
    SolverConfig,
    TimeConfig,
)
from plm_data.core.output import FrameWriter
from plm_data.core.runtime import is_complex_runtime
from plm_data.presets import get_preset
from plm_data.presets.base import RunResult

HAS_DOLFINX_MPC = importlib.util.find_spec("dolfinx_mpc") is not None
_AXIS_SIDE_NAMES = {
    0: ("x-", "x+"),
    1: ("y-", "y+"),
    2: ("z-", "z+"),
}


def constant(value):
    return FieldExpressionConfig(type="constant", params={"value": value})


def scalar_expr(expr_type: str, **params):
    return FieldExpressionConfig(type=expr_type, params=params)


def vector_expr(**components):
    return FieldExpressionConfig(components=components)


def vector_zero():
    return FieldExpressionConfig(type="zero", params={})


def output_fields(**modes):
    return {name: OutputSelectionConfig(mode=mode) for name, mode in modes.items()}


def boundary_field_config(
    boundary_conditions: dict[str, BoundaryConditionConfig],
    *,
    periodic_axes: tuple[int, ...] = (),
) -> BoundaryFieldConfig:
    sides = {
        name: [bc] if isinstance(bc, BoundaryConditionConfig) else list(bc)
        for name, bc in boundary_conditions.items()
    }
    for axis in periodic_axes:
        minus, plus = _AXIS_SIDE_NAMES[axis]
        sides[minus] = [BoundaryConditionConfig(type="periodic", pair_with=plus)]
        sides[plus] = [BoundaryConditionConfig(type="periodic", pair_with=minus)]
    return BoundaryFieldConfig(sides=sides)


def rectangle_domain(
    *,
    mesh_resolution: tuple[int, int] = (8, 8),
) -> DomainConfig:
    return DomainConfig(
        type="rectangle",
        params={"size": [1.0, 1.0], "mesh_resolution": list(mesh_resolution)},
    )


def direct_solver_config() -> SolverConfig:
    return SolverConfig(options={"ksp_type": "preonly", "pc_type": "lu"})


def flow_solver_config() -> SolverConfig:
    return SolverConfig(
        options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": "80",
            "mat_mumps_icntl_24": "1",
            "mat_mumps_icntl_25": "0",
            "ksp_error_if_not_converged": "1",
        }
    )


def cahn_hilliard_solver_config() -> SolverConfig:
    return SolverConfig(
        options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "none",
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
    )


def make_scalar_preset_config(
    tmp_path: Path,
    *,
    preset: str,
    parameters: dict[str, float],
    boundary_conditions: dict[str, BoundaryConditionConfig],
    source: FieldExpressionConfig,
    initial_condition: FieldExpressionConfig | None = None,
    periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, int] = (6, 6),
    output_resolution: tuple[int, int] = (4, 4),
    solver: SolverConfig | None = None,
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    return SimulationConfig(
        preset=preset,
        parameters=parameters,
        domain=rectangle_domain(mesh_resolution=mesh_resolution),
        inputs={
            "u": InputConfig(
                source=source,
                initial_condition=initial_condition,
            )
        },
        boundary_conditions={
            "u": boundary_field_config(
                boundary_conditions,
                periodic_axes=periodic_axes,
            )
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=output_fields(u="scalar"),
        ),
        solver=solver or direct_solver_config(),
        time=time,
        seed=seed,
    )


def make_flow_preset_config(
    tmp_path: Path,
    *,
    preset: str,
    parameters: dict[str, float],
    boundary_conditions: dict[str, BoundaryConditionConfig],
    source: FieldExpressionConfig,
    initial_condition: FieldExpressionConfig | None = None,
    periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, int] = (8, 8),
    output_resolution: tuple[int, int] = (4, 4),
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    return SimulationConfig(
        preset=preset,
        parameters=parameters,
        domain=rectangle_domain(mesh_resolution=mesh_resolution),
        inputs={
            "velocity": InputConfig(
                source=source,
                initial_condition=initial_condition,
            )
        },
        boundary_conditions={
            "velocity": boundary_field_config(
                boundary_conditions,
                periodic_axes=periodic_axes,
            )
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=output_fields(velocity="components", pressure="scalar"),
        ),
        solver=flow_solver_config(),
        time=time,
        seed=seed,
    )


def make_cahn_hilliard_config(
    tmp_path: Path,
    *,
    initial_condition: FieldExpressionConfig,
    periodic_axes: tuple[int, ...] = (0, 1),
    mesh_resolution: tuple[int, int] = (6, 6),
    output_resolution: tuple[int, int] = (4, 4),
    seed: int | None = 42,
) -> SimulationConfig:
    return SimulationConfig(
        preset="cahn_hilliard",
        parameters={
            "lmbda": 0.01,
            "barrier_height": 100.0,
            "mobility": 1.0,
            "theta": 0.5,
        },
        domain=rectangle_domain(mesh_resolution=mesh_resolution),
        inputs={"c": InputConfig(initial_condition=initial_condition)},
        boundary_conditions={
            "c": boundary_field_config({}, periodic_axes=periodic_axes)
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(c="scalar"),
        ),
        solver=cahn_hilliard_solver_config(),
        time=TimeConfig(dt=5e-6, t_end=5e-6),
        seed=seed,
    )


def make_maxwell_config(
    tmp_path: Path,
    *,
    boundary_conditions: dict[str, BoundaryConditionConfig],
    source: FieldExpressionConfig,
    periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, int] = (6, 6),
    output_resolution: tuple[int, int] = (4, 4),
    seed: int | None = 42,
) -> SimulationConfig:
    return SimulationConfig(
        preset="maxwell",
        parameters={
            "epsilon_r": 1.0,
            "mu_r": 1.0,
            "k0": 8.0,
            "source_amplitude": 1.0,
        },
        domain=rectangle_domain(mesh_resolution=mesh_resolution),
        inputs={
            "electric_field": InputConfig(
                source=source,
            )
        },
        boundary_conditions={
            "electric_field": boundary_field_config(
                boundary_conditions,
                periodic_axes=periodic_axes,
            )
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=1,
            formats=["numpy"],
            fields=output_fields(electric_field="components"),
        ),
        solver=direct_solver_config(),
        seed=seed,
    )


def make_maxwell_pulse_config(
    tmp_path: Path,
    *,
    boundary_conditions: dict[str, BoundaryConditionConfig],
    source: FieldExpressionConfig,
    initial_condition: FieldExpressionConfig,
    periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, int] = (6, 6),
    output_resolution: tuple[int, int] = (4, 4),
    seed: int | None = 42,
) -> SimulationConfig:
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
        domain=rectangle_domain(mesh_resolution=mesh_resolution),
        inputs={
            "electric_field": InputConfig(
                source=source,
                initial_condition=initial_condition,
            )
        },
        boundary_conditions={
            "electric_field": boundary_field_config(
                boundary_conditions,
                periodic_axes=periodic_axes,
            )
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(electric_field="components"),
        ),
        solver=direct_solver_config(),
        time=TimeConfig(dt=0.02, t_end=0.02),
        seed=seed,
    )


def run_preset(config: SimulationConfig) -> tuple[RunResult, Path]:
    preset = get_preset(config.preset)
    output_dir = config.output.path / preset.spec.category / preset.spec.name
    writer = FrameWriter(output_dir, config, preset.spec)
    result = preset.build_problem(config).run(writer)
    writer.finalize()
    return result, output_dir


def load_expected_output_arrays(
    config: SimulationConfig,
    output_dir: Path,
) -> dict[str, NDArray[np.generic]]:
    preset = get_preset(config.preset)
    output_modes = {
        output_name: selection.mode
        for output_name, selection in config.output.fields.items()
    }
    arrays: dict[str, NDArray[np.generic]] = {}
    for concrete in preset.spec.expected_outputs(output_modes, config.domain.dimension):
        arrays[concrete.name] = np.load(output_dir / f"{concrete.name}.npy")
    return arrays


def assert_expected_output_arrays(
    config: SimulationConfig,
    output_dir: Path,
) -> dict[str, NDArray[np.generic]]:
    arrays = load_expected_output_arrays(config, output_dir)
    expected_shape = (config.output.num_frames, *config.output.resolution)
    for array in arrays.values():
        assert array.shape == expected_shape
        assert np.all(np.isfinite(array))
    return arrays


def assert_nontrivial(array: NDArray[np.generic], *, threshold: float = 1e-8) -> None:
    assert np.max(np.abs(array)) > threshold


def assert_periodic_axis(
    array: NDArray[np.generic],
    axis: int,
    *,
    frame: int = -1,
    atol: float = 1e-4,
) -> None:
    start = [frame] + [slice(None)] * (array.ndim - 1)
    end = [frame] + [slice(None)] * (array.ndim - 1)
    start[axis + 1] = 0
    end[axis + 1] = -1
    assert np.allclose(array[tuple(start)], array[tuple(end)], atol=atol)


def skip_without_mpc() -> str | None:
    if HAS_DOLFINX_MPC:
        return None
    return "periodic coverage requires dolfinx_mpc"


def skip_without_complex_runtime() -> str | None:
    if is_complex_runtime():
        return None
    return "harmonic Maxwell requires a complex-valued runtime"


def _never_skip() -> str | None:
    return None


@dataclass(frozen=True)
class RuntimePresetCase:
    """A single runtime capability scenario for a preset family."""

    name: str
    make_config: Callable[[Path], SimulationConfig]
    assert_result: Callable[[SimulationConfig, RunResult, Path], None] | None = None
    expected_error: type[BaseException] | None = None
    expected_error_match: str | None = None
    skip_reason: Callable[[], str | None] = _never_skip

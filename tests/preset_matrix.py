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
from plm_data.core.solver_strategies import (
    CONSTANT_LHS_BLOCK_DIRECT,
    CONSTANT_LHS_CURL_DIRECT,
    CONSTANT_LHS_SCALAR_NONSYMMETRIC,
    CONSTANT_LHS_SCALAR_SPD,
    NONLINEAR_MIXED_DIRECT,
    STATIONARY_INDEFINITE_DIRECT,
    STATIONARY_SCALAR_SPD,
    STEADY_SADDLE_POINT,
    TRANSIENT_EXPLICIT,
    TRANSIENT_MIXED_DIRECT,
    TRANSIENT_SADDLE_POINT,
)
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


def box_domain(
    *,
    mesh_resolution: tuple[int, int, int] = (6, 6, 4),
) -> DomainConfig:
    return DomainConfig(
        type="box",
        params={"size": [1.0, 1.0, 1.0], "mesh_resolution": list(mesh_resolution)},
    )


def solver_config(
    strategy: str,
    *,
    serial: dict[str, str],
    mpi: dict[str, str] | None = None,
) -> SolverConfig:
    return SolverConfig(
        strategy=strategy,
        serial=serial,
        mpi=serial if mpi is None else mpi,
    )


def direct_solver_config(strategy: str = STATIONARY_SCALAR_SPD) -> SolverConfig:
    mpi = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_14": "80",
        "mat_mumps_icntl_24": "1",
        "mat_mumps_icntl_25": "0",
        "ksp_error_if_not_converged": "1",
    }
    if strategy in {STATIONARY_SCALAR_SPD, CONSTANT_LHS_SCALAR_SPD}:
        mpi = {
            "ksp_type": "cg",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": "1.0e-10",
            "ksp_error_if_not_converged": "1",
        }
    elif strategy == CONSTANT_LHS_SCALAR_NONSYMMETRIC:
        mpi = {
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": "1.0e-9",
            "ksp_error_if_not_converged": "1",
        }
    return solver_config(
        strategy,
        serial={"ksp_type": "preonly", "pc_type": "lu"},
        mpi=mpi,
    )


def flow_solver_config(
    strategy: str = STEADY_SADDLE_POINT,
) -> SolverConfig:
    serial = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_14": "80",
        "mat_mumps_icntl_24": "1",
        "mat_mumps_icntl_25": "0",
        "ksp_error_if_not_converged": "1",
    }
    mpi = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_14": "80",
        "mat_mumps_icntl_24": "1",
        "mat_mumps_icntl_25": "0",
        "ksp_error_if_not_converged": "1",
    }
    if strategy == STEADY_SADDLE_POINT:
        mpi = {
            "ksp_type": "minres",
            "ksp_rtol": "1.0e-9",
            "ksp_error_if_not_converged": "1",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "additive",
            "fieldsplit_velocity_0_ksp_type": "preonly",
            "fieldsplit_velocity_0_pc_type": "gamg",
            "fieldsplit_pressure_1_ksp_type": "preonly",
            "fieldsplit_pressure_1_pc_type": "jacobi",
        }
    elif strategy == TRANSIENT_SADDLE_POINT:
        mpi = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": "80",
            "mat_mumps_icntl_24": "1",
            "mat_mumps_icntl_25": "0",
            "ksp_error_if_not_converged": "1",
        }
    return solver_config(
        strategy,
        serial=serial,
        mpi=mpi,
    )


def cahn_hilliard_solver_config() -> SolverConfig:
    return nonlinear_mixed_direct_solver_config()


def nonlinear_mixed_direct_solver_config() -> SolverConfig:
    return solver_config(
        NONLINEAR_MIXED_DIRECT,
        serial={
            "snes_type": "newtonls",
            "snes_linesearch_type": "none",
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        mpi={
            "snes_type": "newtonls",
            "snes_linesearch_type": "none",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": "1",
        },
    )


def explicit_solver_config() -> SolverConfig:
    return solver_config(
        TRANSIENT_EXPLICIT,
        serial={},
        mpi={},
    )


def make_scalar_preset_config(
    tmp_path: Path,
    *,
    preset: str,
    parameters: dict[str, float],
    coefficients: dict[str, FieldExpressionConfig] | None = None,
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
        solver=solver
        or direct_solver_config(
            {
                "advection": CONSTANT_LHS_SCALAR_NONSYMMETRIC,
                "poisson": STATIONARY_SCALAR_SPD,
                "heat": CONSTANT_LHS_SCALAR_SPD,
                "helmholtz": STATIONARY_INDEFINITE_DIRECT,
            }[preset]
        ),
        time=time,
        seed=seed,
        coefficients={} if coefficients is None else coefficients,
    )


def make_advection_config(
    tmp_path: Path,
    *,
    gdim: int,
    velocity: FieldExpressionConfig,
    diffusivity: FieldExpressionConfig,
    boundary_conditions: dict[str, BoundaryConditionConfig],
    source: FieldExpressionConfig,
    initial_condition: FieldExpressionConfig,
    periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    solver: SolverConfig | None = None,
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(f"Advection test helper only supports 2D/3D, got {gdim}D")

    return SimulationConfig(
        preset="advection",
        parameters={},
        domain=domain,
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
        solver=solver or direct_solver_config(CONSTANT_LHS_SCALAR_NONSYMMETRIC),
        time=time,
        seed=seed,
        coefficients={
            "velocity": velocity,
            "diffusivity": diffusivity,
        },
    )


def make_darcy_config(
    tmp_path: Path,
    *,
    gdim: int,
    parameters: dict[str, float],
    mobility: FieldExpressionConfig,
    dispersion: FieldExpressionConfig,
    pressure_boundary_conditions: dict[str, BoundaryConditionConfig],
    concentration_boundary_conditions: dict[str, BoundaryConditionConfig],
    pressure_source: FieldExpressionConfig,
    pressure_initial_condition: FieldExpressionConfig,
    concentration_source: FieldExpressionConfig,
    concentration_initial_condition: FieldExpressionConfig,
    periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    solver: SolverConfig | None = None,
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(f"Darcy test helper only supports 2D/3D, got {gdim}D")

    return SimulationConfig(
        preset="darcy",
        parameters=parameters,
        domain=domain,
        inputs={
            "pressure": InputConfig(
                source=pressure_source,
                initial_condition=pressure_initial_condition,
            ),
            "concentration": InputConfig(
                source=concentration_source,
                initial_condition=concentration_initial_condition,
            ),
        },
        boundary_conditions={
            "pressure": boundary_field_config(
                pressure_boundary_conditions,
                periodic_axes=periodic_axes,
            ),
            "concentration": boundary_field_config(
                concentration_boundary_conditions,
                periodic_axes=periodic_axes,
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=output_fields(
                pressure="scalar",
                concentration="scalar",
                velocity="components",
                speed="scalar",
            ),
        ),
        solver=solver or direct_solver_config(CONSTANT_LHS_SCALAR_NONSYMMETRIC),
        time=time,
        seed=seed,
        coefficients={
            "mobility": mobility,
            "dispersion": dispersion,
        },
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
        solver=flow_solver_config(
            STEADY_SADDLE_POINT
            if preset == "stokes"
            else TRANSIENT_SADDLE_POINT
            if not periodic_axes
            else TRANSIENT_MIXED_DIRECT
        ),
        time=time,
        seed=seed,
    )


def make_burgers_config(
    tmp_path: Path,
    *,
    gdim: int,
    parameters: dict[str, float],
    boundary_conditions: dict[str, BoundaryConditionConfig],
    source: FieldExpressionConfig,
    initial_condition: FieldExpressionConfig,
    periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    solver: SolverConfig | None = None,
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(f"Burgers test helper only supports 2D/3D, got {gdim}D")

    return SimulationConfig(
        preset="burgers",
        parameters=parameters,
        domain=domain,
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
            fields=output_fields(velocity="components"),
        ),
        solver=solver or cahn_hilliard_solver_config(),
        time=time,
        seed=seed,
    )


def make_thermal_convection_config(
    tmp_path: Path,
    *,
    gdim: int,
    parameters: dict[str, float],
    velocity_boundary_conditions: dict[str, BoundaryConditionConfig],
    temperature_boundary_conditions: dict[str, BoundaryConditionConfig],
    velocity_source: FieldExpressionConfig,
    velocity_initial_condition: FieldExpressionConfig,
    temperature_source: FieldExpressionConfig,
    temperature_initial_condition: FieldExpressionConfig,
    periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(
            f"Thermal convection test helper only supports 2D/3D, got {gdim}D"
        )

    return SimulationConfig(
        preset="thermal_convection",
        parameters=parameters,
        domain=domain,
        inputs={
            "velocity": InputConfig(
                source=velocity_source,
                initial_condition=velocity_initial_condition,
            ),
            "temperature": InputConfig(
                source=temperature_source,
                initial_condition=temperature_initial_condition,
            ),
        },
        boundary_conditions={
            "velocity": boundary_field_config(
                velocity_boundary_conditions,
                periodic_axes=periodic_axes,
            ),
            "temperature": boundary_field_config(
                temperature_boundary_conditions,
                periodic_axes=periodic_axes,
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=output_fields(
                velocity="components",
                pressure="scalar",
                temperature="scalar",
            ),
        ),
        solver=flow_solver_config(TRANSIENT_MIXED_DIRECT),
        time=time,
        seed=seed,
    )


def make_compressible_navier_stokes_config(
    tmp_path: Path,
    *,
    gdim: int,
    parameters: dict[str, float],
    density_boundary_conditions: dict[str, BoundaryConditionConfig],
    velocity_boundary_conditions: dict[str, BoundaryConditionConfig],
    temperature_boundary_conditions: dict[str, BoundaryConditionConfig],
    density_source: FieldExpressionConfig,
    density_initial_condition: FieldExpressionConfig,
    velocity_source: FieldExpressionConfig,
    velocity_initial_condition: FieldExpressionConfig,
    temperature_source: FieldExpressionConfig,
    temperature_initial_condition: FieldExpressionConfig,
    periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    output_modes: dict[str, OutputSelectionConfig] | None = None,
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(
            f"Compressible Navier-Stokes test helper only supports 2D/3D, got {gdim}D"
        )

    selected_output_modes = (
        output_fields(
            density="scalar",
            velocity="components",
            pressure="scalar",
            temperature="scalar",
        )
        if output_modes is None
        else output_modes
    )

    return SimulationConfig(
        preset="compressible_navier_stokes",
        parameters=parameters,
        domain=domain,
        inputs={
            "density": InputConfig(
                source=density_source,
                initial_condition=density_initial_condition,
            ),
            "velocity": InputConfig(
                source=velocity_source,
                initial_condition=velocity_initial_condition,
            ),
            "temperature": InputConfig(
                source=temperature_source,
                initial_condition=temperature_initial_condition,
            ),
        },
        boundary_conditions={
            "density": boundary_field_config(
                density_boundary_conditions,
                periodic_axes=periodic_axes,
            ),
            "velocity": boundary_field_config(
                velocity_boundary_conditions,
                periodic_axes=periodic_axes,
            ),
            "temperature": boundary_field_config(
                temperature_boundary_conditions,
                periodic_axes=periodic_axes,
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=selected_output_modes,
        ),
        solver=nonlinear_mixed_direct_solver_config(),
        time=time,
        seed=seed,
    )


def make_compressible_euler_config(
    tmp_path: Path,
    *,
    parameters: dict[str, float],
    state_boundary_conditions: dict[str, BoundaryConditionConfig],
    density_initial_condition: FieldExpressionConfig,
    velocity_initial_condition: FieldExpressionConfig,
    pressure_initial_condition: FieldExpressionConfig,
    periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, int] = (8, 8),
    output_resolution: tuple[int, int] = (4, 4),
    output_modes: dict[str, OutputSelectionConfig] | None = None,
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    selected_output_modes = (
        output_fields(
            density="scalar",
            velocity="components",
            pressure="scalar",
            total_energy="scalar",
        )
        if output_modes is None
        else output_modes
    )

    return SimulationConfig(
        preset="compressible_euler",
        parameters=parameters,
        domain=rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        ),
        inputs={
            "density": InputConfig(initial_condition=density_initial_condition),
            "velocity": InputConfig(initial_condition=velocity_initial_condition),
            "pressure": InputConfig(initial_condition=pressure_initial_condition),
        },
        boundary_conditions={
            "state": boundary_field_config(
                state_boundary_conditions,
                periodic_axes=periodic_axes,
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=selected_output_modes,
        ),
        solver=explicit_solver_config(),
        time=time,
        seed=seed,
    )


def make_mhd_config(
    tmp_path: Path,
    *,
    gdim: int,
    parameters: dict[str, float],
    velocity_boundary_conditions: dict[str, BoundaryConditionConfig],
    magnetic_boundary_conditions: dict[str, BoundaryConditionConfig],
    velocity_source: FieldExpressionConfig,
    velocity_initial_condition: FieldExpressionConfig,
    magnetic_source: FieldExpressionConfig,
    magnetic_initial_condition: FieldExpressionConfig,
    velocity_periodic_axes: tuple[int, ...] = (),
    magnetic_periodic_axes: tuple[int, ...] | None = None,
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    output_modes: dict[str, OutputSelectionConfig] | None = None,
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(f"MHD test helper only supports 2D/3D, got {gdim}D")

    selected_output_modes = (
        output_fields(
            velocity="components",
            pressure="scalar",
            magnetic_field="components",
            magnetic_constraint="scalar",
        )
        if output_modes is None
        else output_modes
    )

    return SimulationConfig(
        preset="mhd",
        parameters=parameters,
        domain=domain,
        inputs={
            "velocity": InputConfig(
                source=velocity_source,
                initial_condition=velocity_initial_condition,
            ),
            "magnetic_field": InputConfig(
                source=magnetic_source,
                initial_condition=magnetic_initial_condition,
            ),
        },
        boundary_conditions={
            "velocity": boundary_field_config(
                velocity_boundary_conditions,
                periodic_axes=velocity_periodic_axes,
            ),
            "magnetic_field": boundary_field_config(
                magnetic_boundary_conditions,
                periodic_axes=(
                    velocity_periodic_axes
                    if magnetic_periodic_axes is None
                    else magnetic_periodic_axes
                ),
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=selected_output_modes,
        ),
        solver=flow_solver_config(TRANSIENT_MIXED_DIRECT),
        time=time,
        seed=seed,
    )


def make_kuramoto_sivashinsky_config(
    tmp_path: Path,
    *,
    initial_condition: FieldExpressionConfig,
    periodic_axes: tuple[int, ...] = (0, 1),
    mesh_resolution: tuple[int, int] = (6, 6),
    output_resolution: tuple[int, int] = (4, 4),
    seed: int | None = 42,
) -> SimulationConfig:
    return SimulationConfig(
        preset="kuramoto_sivashinsky",
        parameters={
            "theta": 0.5,
            "hyperdiffusion": 1.0,
            "anti_diffusion": 1.0,
            "nonlinear_strength": 1.0,
            "damping": 0.03,
            "advection_x": 0.0,
            "advection_y": 0.0,
            "advection_z": 0.0,
        },
        domain=rectangle_domain(mesh_resolution=mesh_resolution),
        inputs={"u": InputConfig(initial_condition=initial_condition)},
        boundary_conditions={
            "u": boundary_field_config({}, periodic_axes=periodic_axes)
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(u="scalar", v="scalar"),
        ),
        solver=nonlinear_mixed_direct_solver_config(),
        time=TimeConfig(dt=0.05, t_end=0.05),
        seed=seed,
    )


def make_swift_hohenberg_config(
    tmp_path: Path,
    *,
    gdim: int,
    initial_condition: FieldExpressionConfig,
    velocity: FieldExpressionConfig,
    boundary_conditions: dict[str, BoundaryConditionConfig],
    periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (6, 6),
    output_resolution: tuple[int, ...] = (4, 4),
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(
            f"Swift-Hohenberg test helper only supports 2D/3D, got {gdim}D"
        )

    return SimulationConfig(
        preset="swift_hohenberg",
        parameters={
            "r": -0.28,
            "q0": 1.0,
            "alpha": 1.6,
            "beta": -1.0,
            "gamma": -1.0,
            "theta": 0.5,
        },
        domain=domain,
        inputs={"u": InputConfig(initial_condition=initial_condition)},
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
        solver=nonlinear_mixed_direct_solver_config(),
        time=time or TimeConfig(dt=0.1, t_end=0.1),
        seed=seed,
        coefficients={"velocity": velocity},
    )


def make_zakharov_kuznetsov_config(
    tmp_path: Path,
    *,
    initial_condition: FieldExpressionConfig,
    periodic_axes: tuple[int, ...] = (0, 1),
    mesh_resolution: tuple[int, int] = (6, 6),
    output_resolution: tuple[int, int] = (4, 4),
    seed: int | None = 42,
) -> SimulationConfig:
    return SimulationConfig(
        preset="zakharov_kuznetsov",
        parameters={"alpha": 6.0, "theta": 0.5},
        domain=rectangle_domain(mesh_resolution=mesh_resolution),
        inputs={"u": InputConfig(initial_condition=initial_condition)},
        boundary_conditions={
            "u": boundary_field_config({}, periodic_axes=periodic_axes)
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(u="scalar"),
        ),
        solver=nonlinear_mixed_direct_solver_config(),
        time=TimeConfig(dt=0.01, t_end=0.01),
        seed=seed,
    )


def make_cgl_config(
    tmp_path: Path,
    *,
    u_initial_condition: FieldExpressionConfig,
    v_initial_condition: FieldExpressionConfig,
    periodic_axes: tuple[int, ...] = (0, 1),
    mesh_resolution: tuple[int, int] = (6, 6),
    output_resolution: tuple[int, int] = (4, 4),
    seed: int | None = 42,
) -> SimulationConfig:
    return SimulationConfig(
        preset="cgl",
        parameters={
            "D_r": 1.0,
            "D_i": 2.0,
            "a_r": 1.0,
            "a_i": 0.0,
            "b_r": -1.0,
            "b_i": 1.2,
            "theta": 0.5,
        },
        domain=rectangle_domain(mesh_resolution=mesh_resolution),
        inputs={
            "u": InputConfig(initial_condition=u_initial_condition),
            "v": InputConfig(initial_condition=v_initial_condition),
        },
        boundary_conditions={
            "u": boundary_field_config({}, periodic_axes=periodic_axes),
            "v": boundary_field_config({}, periodic_axes=periodic_axes),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(u="scalar", v="scalar", amplitude="scalar"),
        ),
        solver=nonlinear_mixed_direct_solver_config(),
        time=TimeConfig(dt=0.1, t_end=0.1),
        seed=seed,
    )


def make_schrodinger_config(
    tmp_path: Path,
    *,
    gdim: int,
    u_initial_condition: FieldExpressionConfig,
    v_initial_condition: FieldExpressionConfig,
    u_boundary_conditions: dict[str, BoundaryConditionConfig],
    v_boundary_conditions: dict[str, BoundaryConditionConfig],
    u_periodic_axes: tuple[int, ...] = (),
    v_periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    time: TimeConfig | None = None,
    seed: int | None = 42,
    potential: FieldExpressionConfig | None = None,
    parameters: dict[str, float] | None = None,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(f"Schrodinger test helper only supports 2D/3D, got {gdim}D")

    if parameters is None:
        parameters = {"D": 0.05, "theta": 0.5}
    if time is None:
        time = TimeConfig(dt=0.01, t_end=0.01)

    return SimulationConfig(
        preset="schrodinger",
        parameters=parameters,
        domain=domain,
        inputs={
            "u": InputConfig(initial_condition=u_initial_condition),
            "v": InputConfig(initial_condition=v_initial_condition),
        },
        boundary_conditions={
            "u": boundary_field_config(
                u_boundary_conditions,
                periodic_axes=u_periodic_axes,
            ),
            "v": boundary_field_config(
                v_boundary_conditions,
                periodic_axes=v_periodic_axes,
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(
                u="scalar",
                v="scalar",
                density="scalar",
                potential="scalar",
            ),
        ),
        solver=direct_solver_config(CONSTANT_LHS_BLOCK_DIRECT),
        time=time,
        seed=seed,
        coefficients={
            "potential": constant(0.0) if potential is None else potential,
        },
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


def make_gray_scott_config(
    tmp_path: Path,
    *,
    gdim: int,
    velocity: FieldExpressionConfig,
    u_initial_condition: FieldExpressionConfig,
    v_initial_condition: FieldExpressionConfig,
    u_boundary_conditions: dict[str, BoundaryConditionConfig],
    v_boundary_conditions: dict[str, BoundaryConditionConfig],
    u_periodic_axes: tuple[int, ...] = (),
    v_periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    solver: SolverConfig | None = None,
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(f"Gray-Scott test helper only supports 2D/3D, got {gdim}D")

    return SimulationConfig(
        preset="gray_scott",
        parameters={
            "Du": 2.0e-5,
            "Dv": 1.0e-5,
            "F": 0.037,
            "k": 0.06,
        },
        domain=domain,
        inputs={
            "u": InputConfig(initial_condition=u_initial_condition),
            "v": InputConfig(initial_condition=v_initial_condition),
        },
        boundary_conditions={
            "u": boundary_field_config(
                u_boundary_conditions,
                periodic_axes=u_periodic_axes,
            ),
            "v": boundary_field_config(
                v_boundary_conditions,
                periodic_axes=v_periodic_axes,
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=output_fields(u="scalar", v="scalar"),
        ),
        solver=solver or direct_solver_config(CONSTANT_LHS_SCALAR_NONSYMMETRIC),
        time=time,
        seed=seed,
        coefficients={"velocity": velocity},
    )


def make_scalar_reaction_diffusion_config(
    tmp_path: Path,
    *,
    preset: str,
    gdim: int,
    parameters: dict[str, float],
    velocity: FieldExpressionConfig,
    initial_condition: FieldExpressionConfig,
    boundary_conditions: dict[str, BoundaryConditionConfig],
    periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    solver: SolverConfig | None = None,
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(
            f"Scalar reaction-diffusion helper only supports 2D/3D, got {gdim}D"
        )

    return SimulationConfig(
        preset=preset,
        parameters=parameters,
        domain=domain,
        inputs={"u": InputConfig(initial_condition=initial_condition)},
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
        solver=solver or nonlinear_mixed_direct_solver_config(),
        time=time,
        seed=seed,
        coefficients={"velocity": velocity},
    )


def make_fisher_kpp_config(
    tmp_path: Path,
    *,
    gdim: int,
    velocity: FieldExpressionConfig,
    initial_condition: FieldExpressionConfig,
    boundary_conditions: dict[str, BoundaryConditionConfig],
    periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    solver: SolverConfig | None = None,
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    return make_scalar_reaction_diffusion_config(
        tmp_path,
        preset="fisher_kpp",
        gdim=gdim,
        parameters={"D": 0.1, "r": 1.0, "K": 1.0},
        velocity=velocity,
        initial_condition=initial_condition,
        boundary_conditions=boundary_conditions,
        periodic_axes=periodic_axes,
        mesh_resolution=mesh_resolution,
        output_resolution=output_resolution,
        solver=solver,
        time=time,
        seed=seed,
    )


def make_bistable_travelling_waves_config(
    tmp_path: Path,
    *,
    gdim: int,
    velocity: FieldExpressionConfig,
    initial_condition: FieldExpressionConfig,
    boundary_conditions: dict[str, BoundaryConditionConfig],
    periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    solver: SolverConfig | None = None,
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    return make_scalar_reaction_diffusion_config(
        tmp_path,
        preset="bistable_travelling_waves",
        gdim=gdim,
        parameters={"D": 1.0, "a": 0.3},
        velocity=velocity,
        initial_condition=initial_condition,
        boundary_conditions=boundary_conditions,
        periodic_axes=periodic_axes,
        mesh_resolution=mesh_resolution,
        output_resolution=output_resolution,
        solver=solver,
        time=time,
        seed=seed,
    )


def make_lorenz_config(
    tmp_path: Path,
    *,
    gdim: int,
    x_initial_condition: FieldExpressionConfig,
    y_initial_condition: FieldExpressionConfig,
    z_initial_condition: FieldExpressionConfig,
    x_boundary_conditions: dict[str, BoundaryConditionConfig],
    y_boundary_conditions: dict[str, BoundaryConditionConfig],
    z_boundary_conditions: dict[str, BoundaryConditionConfig],
    x_periodic_axes: tuple[int, ...] = (),
    y_periodic_axes: tuple[int, ...] = (),
    z_periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(f"Lorenz test helper only supports 2D/3D, got {gdim}D")

    return SimulationConfig(
        preset="lorenz",
        parameters={
            "sigma": 10.0,
            "rho": 28.0,
            "beta": 2.6667,
            "D": 1.0,
        },
        domain=domain,
        inputs={
            "x": InputConfig(initial_condition=x_initial_condition),
            "y": InputConfig(initial_condition=y_initial_condition),
            "z": InputConfig(initial_condition=z_initial_condition),
        },
        boundary_conditions={
            "x": boundary_field_config(
                x_boundary_conditions,
                periodic_axes=x_periodic_axes,
            ),
            "y": boundary_field_config(
                y_boundary_conditions,
                periodic_axes=y_periodic_axes,
            ),
            "z": boundary_field_config(
                z_boundary_conditions,
                periodic_axes=z_periodic_axes,
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=output_fields(x="scalar", y="scalar", z="scalar"),
        ),
        solver=direct_solver_config(CONSTANT_LHS_SCALAR_SPD),
        time=time,
        seed=seed,
    )


def make_van_der_pol_config(
    tmp_path: Path,
    *,
    gdim: int,
    u_initial_condition: FieldExpressionConfig,
    v_initial_condition: FieldExpressionConfig,
    u_boundary_conditions: dict[str, BoundaryConditionConfig],
    v_boundary_conditions: dict[str, BoundaryConditionConfig],
    u_periodic_axes: tuple[int, ...] = (),
    v_periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    time: TimeConfig | None = None,
    seed: int | None = 42,
    parameters: dict[str, float] | None = None,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(f"Van der Pol test helper only supports 2D/3D, got {gdim}D")

    if parameters is None:
        parameters = {
            "Du": 0.2,
            "Dv": 0.02,
            "mu": 4.0,
        }

    return SimulationConfig(
        preset="van_der_pol",
        parameters=parameters,
        domain=domain,
        inputs={
            "u": InputConfig(initial_condition=u_initial_condition),
            "v": InputConfig(initial_condition=v_initial_condition),
        },
        boundary_conditions={
            "u": boundary_field_config(
                u_boundary_conditions,
                periodic_axes=u_periodic_axes,
            ),
            "v": boundary_field_config(
                v_boundary_conditions,
                periodic_axes=v_periodic_axes,
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=output_fields(u="scalar", v="scalar"),
        ),
        solver=direct_solver_config(CONSTANT_LHS_SCALAR_SPD),
        time=time,
        seed=seed,
    )


def make_cyclic_competition_config(
    tmp_path: Path,
    *,
    gdim: int,
    u_initial_condition: FieldExpressionConfig,
    v_initial_condition: FieldExpressionConfig,
    w_initial_condition: FieldExpressionConfig,
    u_boundary_conditions: dict[str, BoundaryConditionConfig],
    v_boundary_conditions: dict[str, BoundaryConditionConfig],
    w_boundary_conditions: dict[str, BoundaryConditionConfig],
    u_periodic_axes: tuple[int, ...] = (),
    v_periodic_axes: tuple[int, ...] = (),
    w_periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    time: TimeConfig | None = None,
    seed: int | None = 42,
    parameters: dict[str, float] | None = None,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(
            f"Cyclic competition test helper only supports 2D/3D, got {gdim}D"
        )

    if parameters is None:
        parameters = {
            "Du": 1.0,
            "Dv": 0.5,
            "Dw": 0.25,
            "a": 0.4,
            "b": 1.2,
        }

    return SimulationConfig(
        preset="cyclic_competition",
        parameters=parameters,
        domain=domain,
        inputs={
            "u": InputConfig(initial_condition=u_initial_condition),
            "v": InputConfig(initial_condition=v_initial_condition),
            "w": InputConfig(initial_condition=w_initial_condition),
        },
        boundary_conditions={
            "u": boundary_field_config(
                u_boundary_conditions,
                periodic_axes=u_periodic_axes,
            ),
            "v": boundary_field_config(
                v_boundary_conditions,
                periodic_axes=v_periodic_axes,
            ),
            "w": boundary_field_config(
                w_boundary_conditions,
                periodic_axes=w_periodic_axes,
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=output_fields(u="scalar", v="scalar", w="scalar"),
        ),
        solver=direct_solver_config(CONSTANT_LHS_SCALAR_SPD),
        time=time,
        seed=seed,
    )


def make_immunotherapy_config(
    tmp_path: Path,
    *,
    gdim: int,
    u_initial_condition: FieldExpressionConfig,
    v_initial_condition: FieldExpressionConfig,
    w_initial_condition: FieldExpressionConfig,
    u_boundary_conditions: dict[str, BoundaryConditionConfig],
    v_boundary_conditions: dict[str, BoundaryConditionConfig],
    w_boundary_conditions: dict[str, BoundaryConditionConfig],
    u_periodic_axes: tuple[int, ...] = (),
    v_periodic_axes: tuple[int, ...] = (),
    w_periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    time: TimeConfig | None = None,
    seed: int | None = 42,
    parameters: dict[str, float] | None = None,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(f"Immunotherapy test helper only supports 2D/3D, got {gdim}D")

    if parameters is None:
        parameters = {
            "Du": 100.0,
            "Dv": 1.0,
            "Dw": 100.0,
            "alpha": 0.1,
            "mu_u": 0.167,
            "rho_u": 0.692,
            "gamma_v": 0.1,
            "rho_w": 2.5,
            "gamma_w": 0.001,
            "mu_w": 55.56,
            "sigma_u": 0.015,
            "Ku": 0.0001,
            "sigma_w": 0.0,
            "Kw": 0.0,
        }

    return SimulationConfig(
        preset="immunotherapy",
        parameters=parameters,
        domain=domain,
        inputs={
            "u": InputConfig(initial_condition=u_initial_condition),
            "v": InputConfig(initial_condition=v_initial_condition),
            "w": InputConfig(initial_condition=w_initial_condition),
        },
        boundary_conditions={
            "u": boundary_field_config(
                u_boundary_conditions,
                periodic_axes=u_periodic_axes,
            ),
            "v": boundary_field_config(
                v_boundary_conditions,
                periodic_axes=v_periodic_axes,
            ),
            "w": boundary_field_config(
                w_boundary_conditions,
                periodic_axes=w_periodic_axes,
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=output_fields(u="scalar", v="scalar", w="scalar"),
        ),
        solver=direct_solver_config(CONSTANT_LHS_SCALAR_SPD),
        time=time,
        seed=seed,
    )


def make_gierer_meinhardt_config(
    tmp_path: Path,
    *,
    gdim: int,
    a_initial_condition: FieldExpressionConfig,
    h_initial_condition: FieldExpressionConfig,
    a_boundary_conditions: dict[str, BoundaryConditionConfig],
    h_boundary_conditions: dict[str, BoundaryConditionConfig],
    a_periodic_axes: tuple[int, ...] = (),
    h_periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    time: TimeConfig | None = None,
    seed: int | None = 42,
    parameters: dict[str, float] | None = None,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(
            f"Gierer-Meinhardt test helper only supports 2D/3D, got {gdim}D"
        )

    if parameters is None:
        parameters = {
            "Da": 0.05,
            "Dh": 2.0,
            "rho_a": 1.0,
            "rho_h": 1.0,
            "mu_a": 1.0,
            "mu_h": 1.0,
            "sigma_a": 0.01,
            "sigma_h": 0.0,
            "tau": 1.0,
        }

    return SimulationConfig(
        preset="gierer_meinhardt",
        parameters=parameters,
        domain=domain,
        inputs={
            "a": InputConfig(initial_condition=a_initial_condition),
            "h": InputConfig(initial_condition=h_initial_condition),
        },
        boundary_conditions={
            "a": boundary_field_config(
                a_boundary_conditions,
                periodic_axes=a_periodic_axes,
            ),
            "h": boundary_field_config(
                h_boundary_conditions,
                periodic_axes=h_periodic_axes,
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=output_fields(a="scalar", h="scalar"),
        ),
        solver=direct_solver_config(CONSTANT_LHS_SCALAR_SPD),
        time=time,
        seed=seed,
    )


def make_fitzhugh_nagumo_config(
    tmp_path: Path,
    *,
    gdim: int,
    u_initial_condition: FieldExpressionConfig,
    v_initial_condition: FieldExpressionConfig,
    u_boundary_conditions: dict[str, BoundaryConditionConfig],
    v_boundary_conditions: dict[str, BoundaryConditionConfig],
    u_periodic_axes: tuple[int, ...] = (),
    v_periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    time: TimeConfig | None = None,
    seed: int | None = 42,
    parameters: dict[str, float] | None = None,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(
            f"FitzHugh-Nagumo test helper only supports 2D/3D, got {gdim}D"
        )

    if parameters is None:
        parameters = {
            "Du": 1.0,
            "Dv": 50.0,
            "tau": 0.1,
            "b": 0.5,
            "a": 0.0,
        }

    return SimulationConfig(
        preset="fitzhugh_nagumo",
        parameters=parameters,
        domain=domain,
        inputs={
            "u": InputConfig(initial_condition=u_initial_condition),
            "v": InputConfig(initial_condition=v_initial_condition),
        },
        boundary_conditions={
            "u": boundary_field_config(
                u_boundary_conditions,
                periodic_axes=u_periodic_axes,
            ),
            "v": boundary_field_config(
                v_boundary_conditions,
                periodic_axes=v_periodic_axes,
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=output_fields(u="scalar", v="scalar"),
        ),
        solver=direct_solver_config(CONSTANT_LHS_SCALAR_SPD),
        time=time,
        seed=seed,
    )


def make_klausmeier_topography_config(
    tmp_path: Path,
    *,
    gdim: int,
    topography: FieldExpressionConfig,
    w_initial_condition: FieldExpressionConfig,
    n_initial_condition: FieldExpressionConfig,
    w_boundary_conditions: dict[str, BoundaryConditionConfig],
    n_boundary_conditions: dict[str, BoundaryConditionConfig],
    w_periodic_axes: tuple[int, ...] = (),
    n_periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    time: TimeConfig | None = None,
    seed: int | None = 42,
    parameters: dict[str, float] | None = None,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(
            f"Klausmeier topography test helper only supports 2D/3D, got {gdim}D"
        )

    if parameters is None:
        parameters = {
            "a": 2.0,
            "m": 0.45,
            "D": 10.0,
            "Dn": 1.0,
            "V": 50.0,
        }

    return SimulationConfig(
        preset="klausmeier_topography",
        parameters=parameters,
        domain=domain,
        inputs={
            "w": InputConfig(initial_condition=w_initial_condition),
            "n": InputConfig(initial_condition=n_initial_condition),
        },
        boundary_conditions={
            "w": boundary_field_config(
                w_boundary_conditions,
                periodic_axes=w_periodic_axes,
            ),
            "n": boundary_field_config(
                n_boundary_conditions,
                periodic_axes=n_periodic_axes,
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=output_fields(w="scalar", n="scalar", topography="scalar"),
        ),
        solver=direct_solver_config(CONSTANT_LHS_SCALAR_SPD),
        time=time,
        seed=seed,
        coefficients={"topography": topography},
    )


def make_superlattice_config(
    tmp_path: Path,
    *,
    gdim: int,
    u_1_initial_condition: FieldExpressionConfig,
    v_1_initial_condition: FieldExpressionConfig,
    u_2_initial_condition: FieldExpressionConfig,
    v_2_initial_condition: FieldExpressionConfig,
    u_1_boundary_conditions: dict[str, BoundaryConditionConfig],
    v_1_boundary_conditions: dict[str, BoundaryConditionConfig],
    u_2_boundary_conditions: dict[str, BoundaryConditionConfig],
    v_2_boundary_conditions: dict[str, BoundaryConditionConfig],
    u_1_periodic_axes: tuple[int, ...] = (),
    v_1_periodic_axes: tuple[int, ...] = (),
    u_2_periodic_axes: tuple[int, ...] = (),
    v_2_periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (8, 8),
    output_resolution: tuple[int, ...] = (4, 4),
    time: TimeConfig | None = None,
    seed: int | None = 42,
    parameters: dict[str, float] | None = None,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(f"Superlattice test helper only supports 2D/3D, got {gdim}D")

    if parameters is None:
        parameters = {
            "D_u1": 4.3,
            "D_v1": 50.0,
            "D_u2": 22.0,
            "D_v2": 660.0,
            "a": 3.0,
            "b": 9.0,
            "c": 15.0,
            "d": 9.0,
            "alpha": 0.15,
        }

    return SimulationConfig(
        preset="superlattice",
        parameters=parameters,
        domain=domain,
        inputs={
            "u_1": InputConfig(initial_condition=u_1_initial_condition),
            "v_1": InputConfig(initial_condition=v_1_initial_condition),
            "u_2": InputConfig(initial_condition=u_2_initial_condition),
            "v_2": InputConfig(initial_condition=v_2_initial_condition),
        },
        boundary_conditions={
            "u_1": boundary_field_config(
                u_1_boundary_conditions,
                periodic_axes=u_1_periodic_axes,
            ),
            "v_1": boundary_field_config(
                v_1_boundary_conditions,
                periodic_axes=v_1_periodic_axes,
            ),
            "u_2": boundary_field_config(
                u_2_boundary_conditions,
                periodic_axes=u_2_periodic_axes,
            ),
            "v_2": boundary_field_config(
                v_2_boundary_conditions,
                periodic_axes=v_2_periodic_axes,
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=output_fields(
                u_1="scalar",
                v_1="scalar",
                u_2="scalar",
                v_2="scalar",
            ),
        ),
        solver=direct_solver_config(CONSTANT_LHS_SCALAR_SPD),
        time=time,
        seed=seed,
    )


def make_shallow_water_config(
    tmp_path: Path,
    *,
    parameters: dict[str, float],
    bathymetry: FieldExpressionConfig,
    initial_height: FieldExpressionConfig,
    initial_velocity: FieldExpressionConfig,
    height_boundary_conditions: dict[str, BoundaryConditionConfig],
    velocity_boundary_conditions: dict[str, BoundaryConditionConfig],
    height_periodic_axes: tuple[int, ...] = (),
    velocity_periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, int] = (8, 8),
    output_resolution: tuple[int, int] = (4, 4),
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    return SimulationConfig(
        preset="shallow_water",
        parameters=parameters,
        domain=rectangle_domain(mesh_resolution=mesh_resolution),
        inputs={
            "height": InputConfig(initial_condition=initial_height),
            "velocity": InputConfig(initial_condition=initial_velocity),
        },
        boundary_conditions={
            "height": boundary_field_config(
                height_boundary_conditions,
                periodic_axes=height_periodic_axes,
            ),
            "velocity": boundary_field_config(
                velocity_boundary_conditions,
                periodic_axes=velocity_periodic_axes,
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=output_fields(height="scalar", velocity="components"),
        ),
        solver=nonlinear_mixed_direct_solver_config(),
        time=time,
        seed=seed,
        coefficients={"bathymetry": bathymetry},
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
            fields=output_fields(
                electric_field_real="components",
                electric_field_imag="components",
            ),
        ),
        solver=direct_solver_config(STATIONARY_INDEFINITE_DIRECT),
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
        solver=direct_solver_config(CONSTANT_LHS_CURL_DIRECT),
        time=TimeConfig(dt=0.02, t_end=0.02),
        seed=seed,
    )


def make_plate_config(
    tmp_path: Path,
    *,
    gdim: int,
    boundary_conditions: dict[str, BoundaryConditionConfig],
    deflection_initial_condition: FieldExpressionConfig,
    velocity_initial_condition: FieldExpressionConfig,
    load_source: FieldExpressionConfig,
    parameters: dict[str, float] | None = None,
    coefficients: dict[str, FieldExpressionConfig] | None = None,
    mesh_resolution: tuple[int, ...] = (6, 6),
    output_resolution: tuple[int, ...] = (4, 4),
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(f"Plate test helper only supports 2D/3D, got {gdim}D")

    return SimulationConfig(
        preset="plate",
        parameters={"theta": 0.5} if parameters is None else parameters,
        domain=domain,
        inputs={
            "deflection": InputConfig(initial_condition=deflection_initial_condition),
            "velocity": InputConfig(initial_condition=velocity_initial_condition),
            "load": InputConfig(source=load_source),
        },
        boundary_conditions={"deflection": boundary_field_config(boundary_conditions)},
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=output_fields(deflection="scalar", velocity="scalar"),
        ),
        solver=direct_solver_config(CONSTANT_LHS_BLOCK_DIRECT),
        time=time,
        seed=seed,
        coefficients=(
            {
                "rho_h": constant(1.0),
                "damping": constant(0.1),
                "rigidity": constant(0.02),
            }
            if coefficients is None
            else coefficients
        ),
    )


def make_elasticity_config(
    tmp_path: Path,
    *,
    gdim: int,
    boundary_conditions: dict[str, BoundaryConditionConfig],
    initial_displacement: FieldExpressionConfig,
    initial_velocity: FieldExpressionConfig,
    forcing: FieldExpressionConfig,
    periodic_axes: tuple[int, ...] = (),
    parameters: dict[str, float] | None = None,
    mesh_resolution: tuple[int, ...] = (6, 6),
    output_resolution: tuple[int, ...] = (4, 4),
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(f"Elasticity test helper only supports 2D/3D, got {gdim}D")

    return SimulationConfig(
        preset="elasticity",
        parameters=(
            {
                "young_modulus": 6.0,
                "poisson_ratio": 0.3,
                "density": 1.0,
                "eta_mass": 0.02,
                "eta_stiffness": 0.002,
            }
            if parameters is None
            else parameters
        ),
        domain=domain,
        inputs={
            "displacement": InputConfig(initial_condition=initial_displacement),
            "velocity": InputConfig(initial_condition=initial_velocity),
            "forcing": InputConfig(source=forcing),
        },
        boundary_conditions={
            "displacement": boundary_field_config(
                boundary_conditions,
                periodic_axes=periodic_axes,
            )
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2 if time is not None else 1,
            formats=["numpy"],
            fields=output_fields(
                displacement="components",
                velocity="components",
                von_mises="scalar",
            ),
        ),
        solver=direct_solver_config(CONSTANT_LHS_BLOCK_DIRECT),
        time=time,
        seed=seed,
    )


def make_wave_config(
    tmp_path: Path,
    *,
    gdim: int,
    boundary_conditions: dict[str, BoundaryConditionConfig],
    initial_displacement: FieldExpressionConfig,
    initial_velocity: FieldExpressionConfig,
    forcing: FieldExpressionConfig,
    periodic_axes: tuple[int, ...] = (),
    mesh_resolution: tuple[int, ...] = (6, 6),
    output_resolution: tuple[int, ...] = (4, 4),
    damping: float = 0.0,
    c_sq: FieldExpressionConfig | None = None,
    time: TimeConfig | None = None,
    seed: int | None = 42,
) -> SimulationConfig:
    if gdim == 2:
        domain = rectangle_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    elif gdim == 3:
        domain = box_domain(
            mesh_resolution=tuple(int(value) for value in mesh_resolution)
        )
    else:
        raise ValueError(f"Wave test helper only supports 2D/3D, got {gdim}D")

    return SimulationConfig(
        preset="wave",
        parameters={"damping": damping},
        domain=domain,
        inputs={
            "u": InputConfig(initial_condition=initial_displacement),
            "v": InputConfig(initial_condition=initial_velocity),
            "forcing": InputConfig(source=forcing),
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
            fields=output_fields(u="scalar", v="scalar"),
        ),
        solver=direct_solver_config(CONSTANT_LHS_SCALAR_SPD),
        time=time,
        seed=seed,
        coefficients={"c_sq": constant(1.0) if c_sq is None else c_sq},
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

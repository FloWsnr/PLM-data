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
from plm_data.core.solver_strategies import (
    CONSTANT_LHS_BLOCK_DIRECT,
    CONSTANT_LHS_CURL_DIRECT,
    CONSTANT_LHS_SCALAR_NONSYMMETRIC,
    CONSTANT_LHS_SCALAR_SPD,
    NONLINEAR_MIXED_DIRECT,
    STATIONARY_INDEFINITE_DIRECT,
    STATIONARY_SCALAR_SPD,
    STEADY_SADDLE_POINT,
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
            "ksp_type": "gmres",
            "ksp_rtol": "1.0e-8",
            "ksp_error_if_not_converged": "1",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "upper",
            "pc_fieldsplit_schur_precondition": "selfp",
            "fieldsplit_velocity_0_ksp_type": "preonly",
            "fieldsplit_velocity_0_pc_type": "gamg",
            "fieldsplit_pressure_1_ksp_type": "preonly",
            "fieldsplit_pressure_1_pc_type": "jacobi",
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
            fields=output_fields(electric_field="components"),
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

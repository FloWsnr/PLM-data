"""Shared helpers for focused runtime tests."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from plm_data.core.output import FrameWriter
from plm_data.core.runtime_config import (
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
from plm_data.core.solver_strategies import (
    CONSTANT_LHS_SCALAR_NONSYMMETRIC,
    CONSTANT_LHS_SCALAR_SPD,
)
from plm_data.pdes import get_pde
from plm_data.pdes.base import RunResult


def constant(value):
    return FieldExpressionConfig(type="constant", params={"value": value})


def scalar_expr(expr_type: str, **params):
    return FieldExpressionConfig(type=expr_type, params=params)


def vector_expr(**components):
    return FieldExpressionConfig(components=components)


def vector_zero():
    return FieldExpressionConfig(
        components={
            "x": constant(0.0),
            "y": constant(0.0),
        }
    )


def output_fields(**modes):
    return {name: OutputSelectionConfig(mode=mode) for name, mode in modes.items()}


def boundary_field_config(
    boundary_conditions: dict[str, BoundaryConditionConfig],
) -> BoundaryFieldConfig:
    return BoundaryFieldConfig(
        sides={
            name: [bc] if isinstance(bc, BoundaryConditionConfig) else list(bc)
            for name, bc in boundary_conditions.items()
        }
    )


def solver_config(
    strategy: str,
    *,
    serial: dict[str, str],
    mpi: dict[str, str] | None = None,
) -> SolverConfig:
    return SolverConfig(
        strategy=strategy, serial=serial, mpi=serial if mpi is None else mpi
    )


def direct_solver_config(strategy: str = CONSTANT_LHS_SCALAR_SPD) -> SolverConfig:
    mpi = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_14": "80",
        "mat_mumps_icntl_24": "1",
        "mat_mumps_icntl_25": "0",
        "ksp_error_if_not_converged": "1",
    }
    if strategy == CONSTANT_LHS_SCALAR_SPD:
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


def rectangle_domain(
    *,
    mesh_resolution: tuple[int, int] = (8, 8),
) -> DomainConfig:
    return DomainConfig(
        type="rectangle",
        params={"size": [1.0, 1.0], "mesh_resolution": list(mesh_resolution)},
    )


def make_heat_config(
    tmp_path: Path,
    *,
    boundary_conditions: dict[str, BoundaryConditionConfig],
    source: FieldExpressionConfig,
    initial_condition: FieldExpressionConfig,
    coefficients: dict[str, FieldExpressionConfig],
    mesh_resolution: tuple[int, int] = (8, 8),
    output_resolution: tuple[int, int] = (4, 4),
    time: TimeConfig = TimeConfig(dt=0.01, t_end=0.01),
) -> SimulationConfig:
    return SimulationConfig(
        pde="heat",
        parameters={},
        domain=rectangle_domain(mesh_resolution=mesh_resolution),
        inputs={
            "u": InputConfig(
                source=source,
                initial_condition=initial_condition,
            )
        },
        boundary_conditions={"u": boundary_field_config(boundary_conditions)},
        output=OutputConfig(
            path=tmp_path,
            resolution=list(output_resolution),
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(u="scalar"),
        ),
        solver=direct_solver_config(CONSTANT_LHS_SCALAR_SPD),
        time=time,
        seed=42,
        coefficients=coefficients,
    )


def run_pde(config: SimulationConfig) -> tuple[RunResult, Path]:
    pde = get_pde(config.pde)
    assert config.output.path is not None
    output_dir = config.output.path / pde.spec.category / pde.spec.name
    writer = FrameWriter(output_dir, config, pde.spec)
    result = pde.build_problem(config).run(writer)
    writer.finalize(result.diagnostics)
    return result, output_dir


def assert_expected_output_arrays(
    config: SimulationConfig,
    output_dir: Path,
) -> dict[str, NDArray[np.generic]]:
    pde = get_pde(config.pde)
    output_modes = {
        name: selection.mode for name, selection in config.output.fields.items()
    }
    arrays: dict[str, NDArray[np.generic]] = {}
    for concrete in pde.spec.expected_outputs(output_modes, config.domain.dimension):
        arrays[concrete.name] = np.load(output_dir / f"{concrete.name}.npy")
    return arrays


def assert_nontrivial(array: NDArray[np.generic], *, threshold: float = 1e-8) -> None:
    assert float(np.max(np.abs(array))) > threshold

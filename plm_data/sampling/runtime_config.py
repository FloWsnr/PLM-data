"""Random in-memory runtime-config generation."""

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from plm_data.boundary_conditions.scenarios import (
    compatible_boundary_scenarios,
    get_boundary_scenario,
)
from plm_data.core.runtime_config import (
    DomainConfig,
    FieldExpressionConfig,
    OutputConfig,
    OutputSelectionConfig,
    SimulationConfig,
    SolverConfig,
    TimeConfig,
)
from plm_data.core.solver_strategies import (
    CONSTANT_LHS_CURL_DIRECT,
    CONSTANT_LHS_BLOCK_DIRECT,
    CONSTANT_LHS_SCALAR_NONSYMMETRIC,
    CONSTANT_LHS_SCALAR_SPD,
    NONLINEAR_MIXED_DIRECT,
    TRANSIENT_MIXED_DIRECT,
    TRANSIENT_SADDLE_POINT,
)
from plm_data.pdes import get_pde
from plm_data.initial_conditions.scenarios import (
    compatible_initial_condition_scenarios,
    get_initial_condition_scenario,
)
from plm_data.sampling.context import SamplingContext
from plm_data.sampling.samplers import (
    attempt_rng,
    choose,
    randint,
    sample_numeric_parameter,
    uniform,
)

DEFAULT_RANDOM_OUTPUT_FORMATS = ("numpy", "gif", "vtk")
MIGRATED_RANDOM_PDES = (
    "advection",
    "wave",
    "plate",
    "schrodinger",
    "burgers",
    "heat",
    "bistable_travelling_waves",
    "brusselator",
    "fitzhugh_nagumo",
    "gray_scott",
    "gierer_meinhardt",
    "schnakenberg",
    "cyclic_competition",
    "immunotherapy",
    "van_der_pol",
    "lorenz",
    "keller_segel",
    "klausmeier_topography",
    "superlattice",
    "cahn_hilliard",
    "cgl",
    "kuramoto_sivashinsky",
    "swift_hohenberg",
    "zakharov_kuznetsov",
    "elasticity",
    "fisher_kpp",
    "darcy",
    "navier_stokes",
    "shallow_water",
    "thermal_convection",
    "maxwell_pulse",
)


@dataclass(frozen=True)
class SampledRuntimeConfig:
    """One concrete random simulation configuration and its identity."""

    config: SimulationConfig
    pde_name: str
    pde_category: str
    domain_name: str
    run_id: str
    attempt: int
    case_name: str
    boundary_scenario_name: str
    initial_condition_scenario_name: str

    def output_dir(self, output_root: str | Path) -> Path:
        """Return the final dataset path for this sampled run."""
        return (
            Path(output_root)
            / self.pde_name
            / self.domain_name
            / f"seed_{self.config.seed}_{self.run_id}"
        )

    def metadata(self) -> dict[str, Any]:
        """Return JSON-compatible random sampling metadata."""
        return {
            "pde": self.pde_name,
            "pde_category": self.pde_category,
            "domain": self.domain_name,
            "seed": self.config.seed,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "case": self.case_name,
            "boundary_scenario": self.boundary_scenario_name,
            "initial_condition_scenario": self.initial_condition_scenario_name,
        }


DomainSampler = Callable[[SamplingContext], DomainConfig]
RandomConfigBuilder = Callable[
    [SamplingContext, DomainConfig, str, str],
    SimulationConfig,
]


@dataclass(frozen=True)
class RandomPDEProfile:
    """A migrated random-run PDE profile with compatible domain samplers."""

    name: str
    pde_name: str
    domain_samplers: dict[str, DomainSampler]
    build: RandomConfigBuilder


def _expr(expr_type: str, **params: Any) -> FieldExpressionConfig:
    return FieldExpressionConfig(type=expr_type, params=params)


def _constant(value: float | str) -> FieldExpressionConfig:
    return _expr("constant", value=value)


def _zero() -> FieldExpressionConfig:
    return _expr("zero")


def _vector_zero() -> FieldExpressionConfig:
    return FieldExpressionConfig(type="zero")


def _vector_expr(**components: FieldExpressionConfig) -> FieldExpressionConfig:
    return FieldExpressionConfig(components=components)


def _output_fields(**modes: str) -> dict[str, OutputSelectionConfig]:
    return {name: OutputSelectionConfig(mode=mode) for name, mode in modes.items()}


def _solver(strategy: str) -> SolverConfig:
    serial_lu = {"ksp_type": "preonly", "pc_type": "lu"}
    mpi_direct = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_14": "80",
        "mat_mumps_icntl_24": "1",
        "mat_mumps_icntl_25": "0",
        "ksp_error_if_not_converged": "1",
    }
    mpi_scalar_spd = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": "1.0e-10",
        "ksp_error_if_not_converged": "1",
    }
    mpi_scalar_nonsymmetric = {
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": "1.0e-9",
        "ksp_error_if_not_converged": "1",
    }
    nonlinear_serial = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "none",
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    nonlinear_mpi = {
        **mpi_direct,
        "snes_type": "newtonls",
        "snes_linesearch_type": "none",
    }
    if strategy == CONSTANT_LHS_SCALAR_SPD:
        return SolverConfig(strategy=strategy, serial=serial_lu, mpi=mpi_scalar_spd)
    if strategy == CONSTANT_LHS_SCALAR_NONSYMMETRIC:
        return SolverConfig(
            strategy=strategy,
            serial=serial_lu,
            mpi=mpi_scalar_nonsymmetric,
        )
    if strategy == CONSTANT_LHS_BLOCK_DIRECT:
        return SolverConfig(strategy=strategy, serial=serial_lu, mpi=mpi_direct)
    if strategy in {
        CONSTANT_LHS_CURL_DIRECT,
        TRANSIENT_MIXED_DIRECT,
        TRANSIENT_SADDLE_POINT,
    }:
        return SolverConfig(strategy=strategy, serial=serial_lu, mpi=mpi_direct)
    if strategy == NONLINEAR_MIXED_DIRECT:
        return SolverConfig(
            strategy=strategy, serial=nonlinear_serial, mpi=nonlinear_mpi
        )
    raise ValueError(f"Unsupported random-run solver strategy '{strategy}'.")


def _rectangle_domain(
    context: SamplingContext,
    *,
    length_range: tuple[float, float],
    height_range: tuple[float, float],
    min_cells_x: int,
    max_cells_x: int,
) -> DomainConfig:
    length = uniform(context, "domain.length", *length_range)
    height = uniform(context, "domain.height", *height_range)
    cells_x = randint(context, "domain.cells_x", min_cells_x, max_cells_x)
    cells_y = max(8, int(round(cells_x * height / length)))
    return DomainConfig(
        type="rectangle",
        params={
            "size": [length, height],
            "mesh_resolution": [cells_x, cells_y],
        },
    )


def _rectangle_domain_sampler(
    *,
    length_range: tuple[float, float],
    height_range: tuple[float, float],
    min_cells_x: int,
    max_cells_x: int,
) -> DomainSampler:
    def _sample(context: SamplingContext) -> DomainConfig:
        return _rectangle_domain(
            context,
            length_range=length_range,
            height_range=height_range,
            min_cells_x=min_cells_x,
            max_cells_x=max_cells_x,
        )

    return _sample


def _disk_domain(context: SamplingContext) -> DomainConfig:
    radius = uniform(context, "domain.radius", 0.85, 1.15)
    return DomainConfig(
        type="disk",
        params={
            "center": [0.0, 0.0],
            "radius": radius,
            "mesh_size": uniform(context, "domain.mesh_size", 0.055, 0.075),
        },
    )


def _annulus_domain(context: SamplingContext) -> DomainConfig:
    inner_radius = uniform(context, "domain.inner_radius", 0.22, 0.38)
    outer_radius = uniform(context, "domain.outer_radius", 0.9, 1.2)
    return DomainConfig(
        type="annulus",
        params={
            "center": [0.0, 0.0],
            "inner_radius": inner_radius,
            "outer_radius": max(outer_radius, inner_radius + 0.45),
            "mesh_size": uniform(context, "domain.mesh_size", 0.055, 0.075),
        },
    )


def _output_config(
    context: SamplingContext,
    *,
    fields: dict[str, OutputSelectionConfig],
    base_resolution: tuple[int, int],
    num_frames: int,
) -> OutputConfig:
    jitter = randint(context, "output.resolution_jitter", -8, 8)
    resolution = [
        max(24, base_resolution[0] + jitter),
        max(24, base_resolution[1] + jitter),
    ]
    return OutputConfig(
        path=None,
        resolution=resolution,
        num_frames=num_frames,
        formats=list(DEFAULT_RANDOM_OUTPUT_FORMATS),
        fields=fields,
    )


def _sample_pde_parameters(
    context: SamplingContext,
    pde_name: str,
) -> dict[str, float]:
    spec = get_pde(pde_name).spec
    parameters: dict[str, float] = {}
    for parameter in spec.parameters:
        parameters[parameter.name] = float(
            sample_numeric_parameter(
                context,
                f"{pde_name}.{parameter.name}",
                parameter,
            )
        )
    return parameters


def _build_heat(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "heat")
    return SimulationConfig(
        pde="heat",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(u="scalar"),
            base_resolution=(56, 56),
            num_frames=9,
        ),
        solver=_solver(CONSTANT_LHS_SCALAR_SPD),
        time=TimeConfig(dt=0.01, t_end=0.08),
        seed=context.seed,
        coefficients={"kappa": _constant(uniform(context, "heat.kappa", 0.006, 0.014))},
    )


def _build_advection(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    speed = uniform(context, "advection.speed", 0.35, 0.75)
    angle = uniform(context, "advection.angle", -0.25, 0.25)
    parameters = _sample_pde_parameters(context, "advection")
    return SimulationConfig(
        pde="advection",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(u="scalar"),
            base_resolution=(64, 44),
            num_frames=8,
        ),
        solver=_solver(CONSTANT_LHS_SCALAR_NONSYMMETRIC),
        time=TimeConfig(dt=0.005, t_end=0.035),
        seed=context.seed,
        coefficients={
            "velocity": _vector_expr(
                x=_constant(speed),
                y=_constant(speed * angle),
            ),
            "diffusivity": _constant(
                uniform(context, "advection.diffusivity", 0.0015, 0.004)
            ),
        },
    )


def _build_wave(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "wave")
    return SimulationConfig(
        pde="wave",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(u="scalar", v="scalar"),
            base_resolution=(58, 50),
            num_frames=8,
        ),
        solver=_solver(CONSTANT_LHS_SCALAR_SPD),
        time=TimeConfig(dt=0.004, t_end=0.032),
        seed=context.seed,
        coefficients={"c_sq": _constant(uniform(context, "wave.c_sq", 0.45, 1.0))},
    )


def _build_plate(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "plate")
    return SimulationConfig(
        pde="plate",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(deflection="scalar", velocity="scalar"),
            base_resolution=(58, 42),
            num_frames=7,
        ),
        solver=_solver(CONSTANT_LHS_BLOCK_DIRECT),
        time=TimeConfig(dt=0.003, t_end=0.018),
        seed=context.seed,
        coefficients={
            "rho_h": _constant(uniform(context, "plate.rho_h", 0.9, 1.2)),
            "damping": _constant(uniform(context, "plate.damping", 0.04, 0.12)),
            "rigidity": _constant(uniform(context, "plate.rigidity", 0.012, 0.035)),
        },
    )


def _build_schrodinger(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "schrodinger")
    return SimulationConfig(
        pde="schrodinger",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(
                u="scalar",
                v="scalar",
                density="scalar",
                potential="scalar",
            ),
            base_resolution=(64, 44),
            num_frames=7,
        ),
        solver=_solver(CONSTANT_LHS_BLOCK_DIRECT),
        time=TimeConfig(dt=0.003, t_end=0.018),
        seed=context.seed,
        coefficients={
            "potential": _constant(
                uniform(context, "schrodinger.potential", -0.08, 0.08)
            )
        },
    )


def _build_burgers(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "burgers")
    return SimulationConfig(
        pde="burgers",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(velocity="components"),
            base_resolution=(60, 46),
            num_frames=7,
        ),
        solver=_solver(NONLINEAR_MIXED_DIRECT),
        time=TimeConfig(dt=0.004, t_end=0.024),
        seed=context.seed,
    )


def _build_bistable_travelling_waves(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "bistable_travelling_waves")
    return SimulationConfig(
        pde="bistable_travelling_waves",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(u="scalar"),
            base_resolution=(64, 44),
            num_frames=7,
        ),
        solver=_solver(NONLINEAR_MIXED_DIRECT),
        time=TimeConfig(dt=0.018, t_end=0.108),
        seed=context.seed,
        coefficients={"velocity": _vector_zero()},
    )


def _build_brusselator(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "brusselator")
    return SimulationConfig(
        pde="brusselator",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(u="scalar", v="scalar"),
            base_resolution=(62, 46),
            num_frames=7,
        ),
        solver=_solver(CONSTANT_LHS_SCALAR_SPD),
        time=TimeConfig(dt=0.004, t_end=0.024),
        seed=context.seed,
    )


def _build_fitzhugh_nagumo(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "fitzhugh_nagumo")
    return SimulationConfig(
        pde="fitzhugh_nagumo",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(u="scalar", v="scalar"),
            base_resolution=(62, 46),
            num_frames=7,
        ),
        solver=_solver(CONSTANT_LHS_SCALAR_SPD),
        time=TimeConfig(dt=0.004, t_end=0.024),
        seed=context.seed,
    )


def _build_gray_scott(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "gray_scott")
    return SimulationConfig(
        pde="gray_scott",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(u="scalar", v="scalar"),
            base_resolution=(62, 46),
            num_frames=7,
        ),
        solver=_solver(CONSTANT_LHS_SCALAR_SPD),
        time=TimeConfig(dt=0.004, t_end=0.024),
        seed=context.seed,
        coefficients={"velocity": _vector_zero()},
    )


def _build_gierer_meinhardt(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "gierer_meinhardt")
    return SimulationConfig(
        pde="gierer_meinhardt",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(a="scalar", h="scalar"),
            base_resolution=(62, 46),
            num_frames=7,
        ),
        solver=_solver(CONSTANT_LHS_SCALAR_SPD),
        time=TimeConfig(dt=0.004, t_end=0.024),
        seed=context.seed,
    )


def _build_schnakenberg(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "schnakenberg")
    return SimulationConfig(
        pde="schnakenberg",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(u="scalar", v="scalar"),
            base_resolution=(62, 46),
            num_frames=7,
        ),
        solver=_solver(CONSTANT_LHS_SCALAR_SPD),
        time=TimeConfig(dt=0.003, t_end=0.018),
        seed=context.seed,
    )


def _build_cyclic_competition(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "cyclic_competition")
    return SimulationConfig(
        pde="cyclic_competition",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(u="scalar", v="scalar", w="scalar"),
            base_resolution=(62, 46),
            num_frames=7,
        ),
        solver=_solver(CONSTANT_LHS_SCALAR_SPD),
        time=TimeConfig(dt=0.004, t_end=0.024),
        seed=context.seed,
    )


def _build_immunotherapy(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "immunotherapy")
    return SimulationConfig(
        pde="immunotherapy",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(u="scalar", v="scalar", w="scalar"),
            base_resolution=(64, 48),
            num_frames=7,
        ),
        solver=_solver(CONSTANT_LHS_SCALAR_SPD),
        time=TimeConfig(dt=0.03, t_end=0.18),
        seed=context.seed,
    )


def _build_van_der_pol(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "van_der_pol")
    return SimulationConfig(
        pde="van_der_pol",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(u="scalar", v="scalar"),
            base_resolution=(60, 60),
            num_frames=7,
        ),
        solver=_solver(CONSTANT_LHS_SCALAR_SPD),
        time=TimeConfig(dt=0.006, t_end=0.036),
        seed=context.seed,
    )


def _build_lorenz(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "lorenz")
    return SimulationConfig(
        pde="lorenz",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(x="scalar", y="scalar", z="scalar"),
            base_resolution=(62, 62),
            num_frames=7,
        ),
        solver=_solver(CONSTANT_LHS_SCALAR_SPD),
        time=TimeConfig(dt=0.0015, t_end=0.009),
        seed=context.seed,
    )


def _build_keller_segel(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "keller_segel")
    return SimulationConfig(
        pde="keller_segel",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(rho="scalar", c="scalar"),
            base_resolution=(62, 46),
            num_frames=7,
        ),
        solver=_solver(CONSTANT_LHS_SCALAR_SPD),
        time=TimeConfig(dt=0.003, t_end=0.018),
        seed=context.seed,
    )


def _build_klausmeier_topography(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "klausmeier_topography")
    return SimulationConfig(
        pde="klausmeier_topography",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(w="scalar", n="scalar", topography="scalar"),
            base_resolution=(64, 44),
            num_frames=7,
        ),
        solver=_solver(CONSTANT_LHS_SCALAR_NONSYMMETRIC),
        time=TimeConfig(dt=0.003, t_end=0.018),
        seed=context.seed,
        coefficients={
            "topography": _expr(
                "sine_waves",
                background=0.0,
                modes=[
                    {
                        "amplitude": uniform(
                            context, "klausmeier.topo.amp0", 0.08, 0.18
                        ),
                        "cycles": [1.0, 0.4],
                        "phase": uniform(
                            context, "klausmeier.topo.phase0", 0.0, 6.283185307179586
                        ),
                        "angle": uniform(context, "klausmeier.topo.angle0", -0.5, 0.5),
                    },
                    {
                        "amplitude": uniform(
                            context, "klausmeier.topo.amp1", -0.08, 0.08
                        ),
                        "cycles": [0.4, 0.9],
                        "phase": uniform(
                            context, "klausmeier.topo.phase1", 0.0, 6.283185307179586
                        ),
                        "angle": uniform(context, "klausmeier.topo.angle1", -0.5, 0.5),
                    },
                ],
            )
        },
    )


def _build_superlattice(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "superlattice")
    return SimulationConfig(
        pde="superlattice",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(
                u_1="scalar",
                v_1="scalar",
                u_2="scalar",
                v_2="scalar",
            ),
            base_resolution=(62, 46),
            num_frames=7,
        ),
        solver=_solver(CONSTANT_LHS_SCALAR_SPD),
        time=TimeConfig(dt=0.003, t_end=0.018),
        seed=context.seed,
    )


def _build_cahn_hilliard(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "cahn_hilliard")
    return SimulationConfig(
        pde="cahn_hilliard",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(c="scalar"),
            base_resolution=(56, 44),
            num_frames=6,
        ),
        solver=_solver(NONLINEAR_MIXED_DIRECT),
        time=TimeConfig(dt=0.0015, t_end=0.006),
        seed=context.seed,
    )


def _build_cgl(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "cgl")
    return SimulationConfig(
        pde="cgl",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(u="scalar", v="scalar", amplitude="scalar"),
            base_resolution=(56, 44),
            num_frames=6,
        ),
        solver=_solver(NONLINEAR_MIXED_DIRECT),
        time=TimeConfig(dt=0.0015, t_end=0.006),
        seed=context.seed,
    )


def _build_kuramoto_sivashinsky(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "kuramoto_sivashinsky")
    return SimulationConfig(
        pde="kuramoto_sivashinsky",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(u="scalar", v="scalar"),
            base_resolution=(58, 48),
            num_frames=6,
        ),
        solver=_solver(NONLINEAR_MIXED_DIRECT),
        time=TimeConfig(dt=0.001, t_end=0.005),
        seed=context.seed,
    )


def _build_swift_hohenberg(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "swift_hohenberg")
    return SimulationConfig(
        pde="swift_hohenberg",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(u="scalar"),
            base_resolution=(58, 48),
            num_frames=6,
        ),
        solver=_solver(NONLINEAR_MIXED_DIRECT),
        time=TimeConfig(dt=0.0015, t_end=0.006),
        seed=context.seed,
        coefficients={"velocity": _vector_zero()},
    )


def _build_zakharov_kuznetsov(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "zakharov_kuznetsov")
    return SimulationConfig(
        pde="zakharov_kuznetsov",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(u="scalar"),
            base_resolution=(58, 42),
            num_frames=6,
        ),
        solver=_solver(NONLINEAR_MIXED_DIRECT),
        time=TimeConfig(dt=0.0015, t_end=0.006),
        seed=context.seed,
    )


def _build_fisher_kpp(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "fisher_kpp")
    return SimulationConfig(
        pde="fisher_kpp",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(u="scalar"),
            base_resolution=(64, 48),
            num_frames=8,
        ),
        solver=_solver(NONLINEAR_MIXED_DIRECT),
        time=TimeConfig(dt=0.02, t_end=0.12),
        seed=context.seed,
        coefficients={"velocity": _vector_zero()},
    )


def _build_elasticity(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "elasticity")
    return SimulationConfig(
        pde="elasticity",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(
                displacement="components",
                velocity="components",
                von_mises="scalar",
            ),
            base_resolution=(64, 36),
            num_frames=7,
        ),
        solver=_solver(CONSTANT_LHS_BLOCK_DIRECT),
        time=TimeConfig(dt=0.004, t_end=0.024),
        seed=context.seed,
    )


def _build_darcy(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    height = domain.params["size"][1]
    parameters = _sample_pde_parameters(context, "darcy")
    return SimulationConfig(
        pde="darcy",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(
                pressure="scalar",
                concentration="scalar",
                velocity="components",
                speed="scalar",
            ),
            base_resolution=(64, 40),
            num_frames=7,
        ),
        solver=_solver(CONSTANT_LHS_SCALAR_NONSYMMETRIC),
        time=TimeConfig(dt=0.005, t_end=0.03),
        seed=context.seed,
        coefficients={
            "mobility": _expr(
                "step",
                value_left=uniform(context, "darcy.mobility.left", 0.12, 0.2),
                value_right=uniform(context, "darcy.mobility.right", 0.22, 0.34),
                x_split=uniform(
                    context, "darcy.mobility.split", 0.35 * height, 0.65 * height
                ),
                axis=1,
            ),
            "dispersion": _constant(uniform(context, "darcy.dispersion", 0.001, 0.002)),
        },
    )


def _build_navier_stokes(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "navier_stokes")
    return SimulationConfig(
        pde="navier_stokes",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(velocity="components", pressure="scalar"),
            base_resolution=(56, 36),
            num_frames=6,
        ),
        solver=_solver(TRANSIENT_MIXED_DIRECT),
        time=TimeConfig(dt=0.004, t_end=0.016),
        seed=context.seed,
    )


def _build_shallow_water(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "shallow_water")
    return SimulationConfig(
        pde="shallow_water",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(height="scalar", velocity="components"),
            base_resolution=(58, 42),
            num_frames=6,
        ),
        solver=_solver(NONLINEAR_MIXED_DIRECT),
        time=TimeConfig(dt=0.0015, t_end=0.006),
        seed=context.seed,
        coefficients={"bathymetry": _constant(0.0)},
    )


def _build_thermal_convection(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "thermal_convection")
    return SimulationConfig(
        pde="thermal_convection",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(
                velocity="components",
                pressure="scalar",
                temperature="scalar",
            ),
            base_resolution=(56, 34),
            num_frames=6,
        ),
        solver=_solver(TRANSIENT_MIXED_DIRECT),
        time=TimeConfig(dt=0.003, t_end=0.012),
        seed=context.seed,
    )


def _build_maxwell_pulse(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    parameters = _sample_pde_parameters(context, "maxwell_pulse")
    return SimulationConfig(
        pde="maxwell_pulse",
        parameters=parameters,
        domain=domain,
        inputs=get_initial_condition_scenario(initial_condition_scenario_name).build(
            context,
            domain,
            parameters,
        ),
        boundary_conditions=get_boundary_scenario(boundary_scenario_name).build(
            context,
            domain,
        ),
        output=_output_config(
            context,
            fields=_output_fields(electric_field="components"),
            base_resolution=(56, 38),
            num_frames=6,
        ),
        solver=_solver(CONSTANT_LHS_CURL_DIRECT),
        time=TimeConfig(dt=0.003, t_end=0.018),
        seed=context.seed,
    )


def _random_rectangle_profile(
    name: str,
    pde_name: str,
    build: RandomConfigBuilder,
    *,
    length_range: tuple[float, float],
    height_range: tuple[float, float],
    min_cells_x: int,
    max_cells_x: int,
) -> RandomPDEProfile:
    return RandomPDEProfile(
        name=name,
        pde_name=pde_name,
        domain_samplers={
            "rectangle": _rectangle_domain_sampler(
                length_range=length_range,
                height_range=height_range,
                min_cells_x=min_cells_x,
                max_cells_x=max_cells_x,
            )
        },
        build=build,
    )


_RANDOM_PROFILES = (
    _random_rectangle_profile(
        "scalar_advection_rectangle",
        "advection",
        _build_advection,
        length_range=(1.2, 1.8),
        height_range=(0.8, 1.2),
        min_cells_x=32,
        max_cells_x=44,
    ),
    _random_rectangle_profile(
        "scalar_wave_rectangle",
        "wave",
        _build_wave,
        length_range=(1.0, 1.4),
        height_range=(0.8, 1.2),
        min_cells_x=30,
        max_cells_x=42,
    ),
    _random_rectangle_profile(
        "scalar_plate_rectangle",
        "plate",
        _build_plate,
        length_range=(1.1, 1.6),
        height_range=(0.75, 1.05),
        min_cells_x=24,
        max_cells_x=34,
    ),
    _random_rectangle_profile(
        "split_schrodinger_rectangle",
        "schrodinger",
        _build_schrodinger,
        length_range=(1.3, 1.8),
        height_range=(0.8, 1.2),
        min_cells_x=28,
        max_cells_x=40,
    ),
    _random_rectangle_profile(
        "vector_burgers_rectangle",
        "burgers",
        _build_burgers,
        length_range=(1.1, 1.6),
        height_range=(0.8, 1.2),
        min_cells_x=28,
        max_cells_x=40,
    ),
    RandomPDEProfile(
        name="scalar_heat_2d",
        pde_name="heat",
        domain_samplers={
            "rectangle": _rectangle_domain_sampler(
                length_range=(0.9, 1.3),
                height_range=(0.8, 1.2),
                min_cells_x=28,
                max_cells_x=40,
            ),
            "disk": _disk_domain,
            "annulus": _annulus_domain,
        },
        build=_build_heat,
    ),
    _random_rectangle_profile(
        "nonlinear_bistable_travelling_waves_rectangle",
        "bistable_travelling_waves",
        _build_bistable_travelling_waves,
        length_range=(2.3, 3.2),
        height_range=(1.4, 2.0),
        min_cells_x=28,
        max_cells_x=40,
    ),
    _random_rectangle_profile(
        "reaction_brusselator_rectangle",
        "brusselator",
        _build_brusselator,
        length_range=(1.8, 2.6),
        height_range=(1.2, 1.8),
        min_cells_x=28,
        max_cells_x=40,
    ),
    _random_rectangle_profile(
        "reaction_fitzhugh_nagumo_rectangle",
        "fitzhugh_nagumo",
        _build_fitzhugh_nagumo,
        length_range=(1.6, 2.2),
        height_range=(1.0, 1.5),
        min_cells_x=28,
        max_cells_x=40,
    ),
    _random_rectangle_profile(
        "reaction_gray_scott_rectangle",
        "gray_scott",
        _build_gray_scott,
        length_range=(1.8, 2.6),
        height_range=(1.2, 1.8),
        min_cells_x=28,
        max_cells_x=40,
    ),
    _random_rectangle_profile(
        "reaction_gierer_meinhardt_rectangle",
        "gierer_meinhardt",
        _build_gierer_meinhardt,
        length_range=(1.6, 2.4),
        height_range=(1.1, 1.7),
        min_cells_x=28,
        max_cells_x=40,
    ),
    _random_rectangle_profile(
        "reaction_schnakenberg_rectangle",
        "schnakenberg",
        _build_schnakenberg,
        length_range=(1.8, 2.6),
        height_range=(1.2, 1.8),
        min_cells_x=28,
        max_cells_x=40,
    ),
    _random_rectangle_profile(
        "reaction_cyclic_competition_rectangle",
        "cyclic_competition",
        _build_cyclic_competition,
        length_range=(1.8, 2.6),
        height_range=(1.2, 1.8),
        min_cells_x=28,
        max_cells_x=40,
    ),
    _random_rectangle_profile(
        "reaction_immunotherapy_rectangle",
        "immunotherapy",
        _build_immunotherapy,
        length_range=(18.0, 24.0),
        height_range=(14.0, 20.0),
        min_cells_x=28,
        max_cells_x=40,
    ),
    _random_rectangle_profile(
        "oscillator_van_der_pol_rectangle",
        "van_der_pol",
        _build_van_der_pol,
        length_range=(3.0, 4.2),
        height_range=(3.0, 4.2),
        min_cells_x=28,
        max_cells_x=40,
    ),
    _random_rectangle_profile(
        "chaotic_lorenz_rectangle",
        "lorenz",
        _build_lorenz,
        length_range=(7.2, 9.5),
        height_range=(7.0, 9.0),
        min_cells_x=28,
        max_cells_x=40,
    ),
    _random_rectangle_profile(
        "chemotaxis_keller_segel_rectangle",
        "keller_segel",
        _build_keller_segel,
        length_range=(1.6, 2.4),
        height_range=(1.1, 1.7),
        min_cells_x=28,
        max_cells_x=40,
    ),
    _random_rectangle_profile(
        "vegetation_klausmeier_topography_rectangle",
        "klausmeier_topography",
        _build_klausmeier_topography,
        length_range=(2.2, 3.0),
        height_range=(1.4, 2.0),
        min_cells_x=28,
        max_cells_x=40,
    ),
    _random_rectangle_profile(
        "reaction_superlattice_rectangle",
        "superlattice",
        _build_superlattice,
        length_range=(1.8, 2.6),
        height_range=(1.2, 1.8),
        min_cells_x=28,
        max_cells_x=40,
    ),
    _random_rectangle_profile(
        "phase_cahn_hilliard_rectangle",
        "cahn_hilliard",
        _build_cahn_hilliard,
        length_range=(1.2, 1.8),
        height_range=(1.0, 1.5),
        min_cells_x=24,
        max_cells_x=34,
    ),
    _random_rectangle_profile(
        "oscillator_cgl_rectangle",
        "cgl",
        _build_cgl,
        length_range=(1.4, 2.0),
        height_range=(1.0, 1.5),
        min_cells_x=24,
        max_cells_x=34,
    ),
    _random_rectangle_profile(
        "chaos_kuramoto_sivashinsky_rectangle",
        "kuramoto_sivashinsky",
        _build_kuramoto_sivashinsky,
        length_range=(4.0, 5.5),
        height_range=(3.2, 4.8),
        min_cells_x=24,
        max_cells_x=34,
    ),
    _random_rectangle_profile(
        "pattern_swift_hohenberg_rectangle",
        "swift_hohenberg",
        _build_swift_hohenberg,
        length_range=(4.0, 5.5),
        height_range=(3.2, 4.8),
        min_cells_x=24,
        max_cells_x=34,
    ),
    _random_rectangle_profile(
        "wave_zakharov_kuznetsov_rectangle",
        "zakharov_kuznetsov",
        _build_zakharov_kuznetsov,
        length_range=(2.5, 3.5),
        height_range=(1.5, 2.2),
        min_cells_x=24,
        max_cells_x=34,
    ),
    _random_rectangle_profile(
        "vector_elasticity_rectangle",
        "elasticity",
        _build_elasticity,
        length_range=(1.3, 1.8),
        height_range=(0.45, 0.7),
        min_cells_x=28,
        max_cells_x=38,
    ),
    _random_rectangle_profile(
        "nonlinear_fisher_kpp_rectangle",
        "fisher_kpp",
        _build_fisher_kpp,
        length_range=(2.6, 3.4),
        height_range=(1.6, 2.2),
        min_cells_x=30,
        max_cells_x=42,
    ),
    _random_rectangle_profile(
        "fluid_darcy_rectangle",
        "darcy",
        _build_darcy,
        length_range=(1.8, 2.4),
        height_range=(0.9, 1.25),
        min_cells_x=32,
        max_cells_x=44,
    ),
    _random_rectangle_profile(
        "fluid_navier_stokes_rectangle",
        "navier_stokes",
        _build_navier_stokes,
        length_range=(1.4, 2.0),
        height_range=(0.8, 1.2),
        min_cells_x=18,
        max_cells_x=28,
    ),
    _random_rectangle_profile(
        "fluid_shallow_water_rectangle",
        "shallow_water",
        _build_shallow_water,
        length_range=(1.6, 2.4),
        height_range=(1.0, 1.6),
        min_cells_x=22,
        max_cells_x=32,
    ),
    _random_rectangle_profile(
        "fluid_thermal_convection_rectangle",
        "thermal_convection",
        _build_thermal_convection,
        length_range=(1.6, 2.2),
        height_range=(0.8, 1.1),
        min_cells_x=18,
        max_cells_x=28,
    ),
    _random_rectangle_profile(
        "electromagnetic_maxwell_pulse_rectangle",
        "maxwell_pulse",
        _build_maxwell_pulse,
        length_range=(1.2, 1.8),
        height_range=(0.8, 1.2),
        min_cells_x=20,
        max_cells_x=30,
    ),
)


def list_random_pde_profiles() -> tuple[RandomPDEProfile, ...]:
    """Return the migrated random-run PDE profiles."""
    return _RANDOM_PROFILES


def list_random_pde_cases() -> tuple[RandomPDEProfile, ...]:
    """Return migrated random-run profiles under the historical API name."""
    return list_random_pde_profiles()


def _run_id(seed: int, attempt: int, pde_name: str, domain_name: str) -> str:
    raw = f"{seed}:{attempt}:{pde_name}:{domain_name}".encode()
    return hashlib.sha1(raw).hexdigest()[:8]


def validate_runtime_config(config: SimulationConfig) -> None:
    """Validate generic constraints for a concrete sampled runtime config."""
    pde = get_pde(config.pde)
    spec = pde.spec
    if config.seed is None:
        raise ValueError("Random runtime configs require an explicit seed.")
    spec.validate_dimension(config.domain.dimension)
    if config.domain.dimension != 2:
        raise ValueError(
            f"Random runtime configs support only 2D domains; got "
            f"{config.domain.dimension}D."
        )
    if config.time is None:
        raise ValueError(f"Random PDE '{config.pde}' must be time-dependent.")
    spec.validate_parameters(config.parameters)
    if set(config.inputs) != set(spec.inputs):
        raise ValueError(
            f"Runtime config inputs for PDE '{config.pde}' must be "
            f"{sorted(spec.inputs)}. Got {sorted(config.inputs)}."
        )
    if set(config.boundary_conditions) != set(spec.boundary_fields):
        raise ValueError(
            f"Runtime config boundary fields for PDE '{config.pde}' must be "
            f"{sorted(spec.boundary_fields)}. Got {sorted(config.boundary_conditions)}."
        )
    if set(config.output.fields) != set(spec.outputs):
        raise ValueError(
            f"Runtime config outputs for PDE '{config.pde}' must be "
            f"{sorted(spec.outputs)}. Got {sorted(config.output.fields)}."
        )
    for output_name, output_spec in spec.outputs.items():
        output_spec.validate_output_mode(config.output.fields[output_name].mode)


def _compatible_domain_names(profile: RandomPDEProfile) -> tuple[str, ...]:
    domain_names: list[str] = []
    for domain_name in profile.domain_samplers:
        if compatible_boundary_scenarios(
            pde_name=profile.pde_name,
            domain_name=domain_name,
        ) and compatible_initial_condition_scenarios(
            pde_name=profile.pde_name,
            domain_name=domain_name,
        ):
            domain_names.append(domain_name)
    if not domain_names:
        raise ValueError(
            f"Random PDE profile '{profile.name}' has no domain with both "
            "compatible boundary and initial-condition scenarios."
        )
    return tuple(domain_names)


def _validate_selection_compatibility(
    *,
    pde_name: str,
    domain_name: str,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> None:
    boundary_scenarios = compatible_boundary_scenarios(
        pde_name=pde_name,
        domain_name=domain_name,
    )
    if boundary_scenario_name not in boundary_scenarios:
        raise ValueError(
            f"Boundary scenario '{boundary_scenario_name}' is not compatible "
            f"with {pde_name}/{domain_name}. Available: "
            f"{sorted(boundary_scenarios)}."
        )

    initial_condition_scenarios = compatible_initial_condition_scenarios(
        pde_name=pde_name,
        domain_name=domain_name,
    )
    if initial_condition_scenario_name not in initial_condition_scenarios:
        raise ValueError(
            "Initial-condition scenario "
            f"'{initial_condition_scenario_name}' is not compatible with "
            f"{pde_name}/{domain_name}. Available: "
            f"{sorted(initial_condition_scenarios)}."
        )


def sample_random_runtime_config(seed: int, attempt: int = 0) -> SampledRuntimeConfig:
    """Sample one concrete 2D time-dependent runtime configuration."""
    root_rng = attempt_rng(seed, attempt, "root")
    context = SamplingContext(seed=seed, attempt=attempt, rng=root_rng)
    profile = choose(context, "pde_profile", _RANDOM_PROFILES)
    context.pde_name = profile.pde_name

    domain_name = choose(context, "domain", _compatible_domain_names(profile))
    context.domain_name = domain_name
    domain = profile.domain_samplers[domain_name](context)
    if domain.type != domain_name:
        raise ValueError(
            f"Random PDE profile '{profile.name}' domain sampler '{domain_name}' "
            f"returned domain type '{domain.type}'."
        )

    boundary_scenarios = compatible_boundary_scenarios(
        pde_name=profile.pde_name,
        domain_name=domain_name,
    )
    initial_condition_scenarios = compatible_initial_condition_scenarios(
        pde_name=profile.pde_name,
        domain_name=domain_name,
    )
    boundary_scenario_name = choose(
        context,
        "boundary_scenario",
        tuple(sorted(boundary_scenarios)),
    )
    initial_condition_scenario_name = choose(
        context,
        "initial_condition_scenario",
        tuple(sorted(initial_condition_scenarios)),
    )

    config = profile.build(
        context,
        domain,
        boundary_scenario_name,
        initial_condition_scenario_name,
    )
    if config.pde != profile.pde_name:
        raise ValueError(
            f"Random PDE profile '{profile.name}' returned config for PDE "
            f"'{config.pde}'."
        )
    if config.domain.type != domain_name:
        raise ValueError(
            f"Random PDE profile '{profile.name}' returned config for domain "
            f"'{config.domain.type}'."
        )
    validate_runtime_config(config)
    _validate_selection_compatibility(
        pde_name=profile.pde_name,
        domain_name=domain_name,
        boundary_scenario_name=boundary_scenario_name,
        initial_condition_scenario_name=initial_condition_scenario_name,
    )

    pde_spec = get_pde(profile.pde_name).spec
    return SampledRuntimeConfig(
        config=config,
        pde_name=profile.pde_name,
        pde_category=pde_spec.category,
        domain_name=domain_name,
        run_id=_run_id(seed, attempt, profile.pde_name, domain_name),
        attempt=attempt,
        case_name=profile.name,
        boundary_scenario_name=boundary_scenario_name,
        initial_condition_scenario_name=initial_condition_scenario_name,
    )

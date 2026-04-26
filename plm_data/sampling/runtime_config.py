"""Random in-memory runtime-config generation."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import hashlib
from pathlib import Path
from typing import Any

from plm_data.boundary_conditions.scenarios import (
    compatible_boundary_scenarios,
    get_boundary_scenario,
)
from plm_data.core.runtime_config import (
    DomainConfig,
    OutputConfig,
    OutputSelectionConfig,
    SimulationConfig,
    SolverConfig,
    TimeConfig,
)
from plm_data.domains.registry import list_domain_specs
from plm_data.core.solver_strategies import (
    CONSTANT_LHS_CURL_DIRECT,
    CONSTANT_LHS_BLOCK_DIRECT,
    CONSTANT_LHS_SCALAR_NONSYMMETRIC,
    CONSTANT_LHS_SCALAR_SPD,
    NONLINEAR_MIXED_DIRECT,
    TRANSIENT_MIXED_DIRECT,
    TRANSIENT_SADDLE_POINT,
)
from plm_data.pdes import get_pde, list_pdes
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
)
from plm_data.sampling.specs import RandomPDEOptions

DEFAULT_RANDOM_OUTPUT_FORMATS = ("numpy", "gif", "vtk")
MIGRATED_RANDOM_PDES = tuple(list_pdes())


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


@dataclass(frozen=True)
class _DomainProfileChoice:
    name: str
    sample: Callable[[SamplingContext, dict[str, Any]], DomainConfig]
    constraints: dict[str, Any]


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


def _output_config(
    context: SamplingContext,
    *,
    fields: dict[str, OutputSelectionConfig],
    base_resolution: tuple[int, int],
    num_frames: int,
    resolution_jitter: tuple[int, int],
) -> OutputConfig:
    jitter = randint(context, "output.resolution_jitter", *resolution_jitter)
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


def _require_random_options(pde_name: str) -> RandomPDEOptions:
    options = get_pde(pde_name).spec.random_options
    if not isinstance(options, RandomPDEOptions):
        raise ValueError(f"PDE '{pde_name}' does not expose random-run options.")
    return options


def _random_pde_names() -> tuple[str, ...]:
    return tuple(
        pde_name
        for pde_name in list_pdes()
        if isinstance(get_pde(pde_name).spec.random_options, RandomPDEOptions)
    )


def _sample_domain_from_profiles(
    context: SamplingContext,
    *,
    domain_name: str,
    choices: tuple[_DomainProfileChoice, ...],
) -> DomainConfig:
    if not choices:
        raise ValueError(f"Domain '{domain_name}' has no random profiles to sample.")
    if len(choices) == 1:
        choice = choices[0]
    else:
        profile_index = randint(
            context,
            f"domain_profile.{domain_name}",
            0,
            len(choices) - 1,
        )
        choice = choices[profile_index]
    context.values[f"domain.{domain_name}.profile"] = choice.name
    return choice.sample(context, choice.constraints)


def _domain_samplers_for_options(
    options: RandomPDEOptions,
) -> dict[str, DomainSampler]:
    choices_by_domain: dict[str, list[_DomainProfileChoice]] = {}
    for domain_name, domain_spec in sorted(list_domain_specs().items()):
        if domain_spec.dimension != 2:
            continue
        for domain_profile in domain_spec.random_profiles:
            if not options.allows_domain_profile(domain_name, domain_profile.name):
                continue
            constraints = options.constraints_for(domain_name, domain_profile.name)
            choices_by_domain.setdefault(domain_name, []).append(
                _DomainProfileChoice(
                    name=domain_profile.name,
                    sample=domain_profile.sample,
                    constraints=constraints,
                )
            )
    return {
        domain_name: partial(
            _sample_domain_from_profiles,
            domain_name=domain_name,
            choices=tuple(choices),
        )
        for domain_name, choices in choices_by_domain.items()
    }


def _build_spec_driven_config(
    context: SamplingContext,
    domain: DomainConfig,
    boundary_scenario_name: str,
    initial_condition_scenario_name: str,
) -> SimulationConfig:
    if context.pde_name is None:
        raise ValueError("Sampling context is missing 'pde_name'.")
    options = _require_random_options(context.pde_name)
    parameters = _sample_pde_parameters(context, context.pde_name)
    coefficients = (
        {}
        if options.coefficient_sampler is None
        else options.coefficient_sampler(context, domain, parameters)
    )
    return SimulationConfig(
        pde=context.pde_name,
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
            fields=_output_fields(**options.output.fields),
            base_resolution=options.output.base_resolution,
            num_frames=options.output.num_frames,
            resolution_jitter=options.output.resolution_jitter,
        ),
        solver=_solver(options.solver_strategy),
        time=TimeConfig(dt=options.time.dt, t_end=options.time.t_end),
        seed=context.seed,
        coefficients=coefficients,
    )


def list_random_pde_profiles() -> tuple[RandomPDEProfile, ...]:
    """Return random-run PDE profiles derived from PDE and domain specs."""
    profiles: list[RandomPDEProfile] = []
    for pde_name in _random_pde_names():
        options = _require_random_options(pde_name)
        profiles.append(
            RandomPDEProfile(
                name=options.case_name,
                pde_name=pde_name,
                domain_samplers=_domain_samplers_for_options(options),
                build=_build_spec_driven_config,
            )
        )
    return tuple(profiles)


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
    profile = choose(context, "pde_profile", list_random_pde_profiles())
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

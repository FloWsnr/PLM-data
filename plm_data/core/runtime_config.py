"""Concrete runtime configuration contracts.

These dataclasses are the in-memory configuration used by sampled random runs
and by PDE runtime objects. YAML parsing and shared-fragment compatibility are
not part of the refactored runtime path.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mpi4py import MPI

from plm_data.domains.validation import (
    infer_domain_dimension,
    validate_domain_params,
)

_GRID_FORMATS = {"numpy", "gif", "video"}


@dataclass
class FieldExpressionConfig:
    """Scalar or component-wise field value configuration."""

    type: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    components: dict[str, "FieldExpressionConfig"] = field(default_factory=dict)

    @property
    def is_componentwise(self) -> bool:
        """Return whether this expression is defined per component."""
        return bool(self.components)


@dataclass
class BoundaryConditionConfig:
    """Configuration for one boundary operator entry."""

    type: str
    value: FieldExpressionConfig | None = None
    pair_with: str | None = None
    operator_parameters: dict[str, Any] = field(default_factory=dict)
    alpha: Any | None = None

    def __post_init__(self) -> None:
        if self.alpha is not None:
            self.operator_parameters = {
                **self.operator_parameters,
                "alpha": self.alpha,
            }


@dataclass
class BoundaryFieldConfig:
    """Configuration for all side conditions of one BC-addressable field."""

    sides: dict[str, list[BoundaryConditionConfig]] = field(default_factory=dict)

    def side_conditions(self, name: str) -> list[BoundaryConditionConfig]:
        """Return configured conditions for one side."""
        if name not in self.sides:
            raise KeyError(f"Unknown boundary side '{name}'")
        return self.sides[name]

    def periodic_pair_keys(self) -> set[frozenset[str]]:
        """Return all active periodic side pairs."""
        pairs: set[frozenset[str]] = set()
        for side, entries in self.sides.items():
            for entry in entries:
                if entry.type == "periodic":
                    if entry.pair_with is None:
                        raise ValueError(
                            f"Periodic boundary on side '{side}' is missing "
                            "'pair_with'."
                        )
                    pairs.add(frozenset({side, entry.pair_with}))
        return pairs

    @property
    def has_periodic(self) -> bool:
        """Return whether any side uses the periodic operator."""
        return bool(self.periodic_pair_keys())


@dataclass
class PeriodicMapConfig:
    """Declarative periodic map for a custom or imported domain."""

    slave: str
    master: str
    matrix: list[list[float]]
    offset: list[float]


@dataclass
class DomainConfig:
    """Domain geometry configuration."""

    type: str
    params: dict[str, Any]
    periodic_maps: dict[str, PeriodicMapConfig] = field(default_factory=dict)
    allow_sampling: bool = False

    def __post_init__(self) -> None:
        validate_domain_params(
            self.type,
            self.params,
            allow_sampling=self.allow_sampling,
        )

    @property
    def dimension(self) -> int:
        """Return the spatial dimension."""
        return infer_domain_dimension(self.type, self.params)


@dataclass
class OutputSelectionConfig:
    """Per-output selection policy."""

    mode: str


@dataclass
class InputConfig:
    """Configuration for one PDE input."""

    source: FieldExpressionConfig | None = None
    initial_condition: FieldExpressionConfig | None = None


@dataclass
class OutputConfig:
    """Output configuration."""

    resolution: list[int]
    num_frames: int
    formats: list[str]
    fields: dict[str, OutputSelectionConfig]
    path: Path | None = None

    @property
    def needs_grid_interpolation(self) -> bool:
        """True if any format requires interpolated numpy arrays."""
        return bool(_GRID_FORMATS & set(self.formats))


@dataclass
class SolverConfig:
    """PETSc solver strategy and explicit serial / MPI option profiles."""

    strategy: str
    serial: dict[str, str]
    mpi: dict[str, str]

    def options_for_size(self, comm_size: int) -> dict[str, str]:
        """Return the active PETSc options for the communicator size."""
        if comm_size > 1:
            return self.mpi
        return self.serial

    def profile_name_for_size(self, comm_size: int) -> str:
        """Return the active solver-profile name for the communicator size."""
        if comm_size > 1:
            return "mpi"
        return "serial"

    @property
    def options(self) -> dict[str, str]:
        """Return the active PETSc options for the current communicator."""
        return self.options_for_size(MPI.COMM_WORLD.size)

    @property
    def profile_name(self) -> str:
        """Return the active profile name for the current communicator."""
        return self.profile_name_for_size(MPI.COMM_WORLD.size)


@dataclass
class TimeConfig:
    """Time-stepping configuration."""

    dt: float
    t_end: float


@dataclass
class StateStochasticConfig:
    """Dynamic stochastic forcing for one state variable."""

    coupling: str
    intensity: float
    offset: float | None = None


@dataclass
class CoefficientSmoothingConfig:
    """Optional diffusion-style smoothing for static random media."""

    pseudo_dt: float
    steps: int


@dataclass
class CoefficientStochasticConfig:
    """Static stochastic overlay for one scalar coefficient."""

    mode: str
    std: float
    smoothing: CoefficientSmoothingConfig | None = None
    clamp_min: float | None = None


@dataclass
class StochasticConfig:
    """Validated stochastic runtime configuration."""

    states: dict[str, StateStochasticConfig] = field(default_factory=dict)
    coefficients: dict[str, CoefficientStochasticConfig] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        """Return whether any stochastic feature is active."""
        return bool(self.states or self.coefficients)


@dataclass(init=False)
class SimulationConfig:
    """Concrete simulation runtime configuration."""

    pde: str
    parameters: dict[str, Any]
    domain: DomainConfig
    inputs: dict[str, InputConfig]
    boundary_conditions: dict[str, BoundaryFieldConfig]
    output: OutputConfig
    solver: SolverConfig
    time: TimeConfig | None = None
    seed: int | None = None
    coefficients: dict[str, FieldExpressionConfig] = field(default_factory=dict)
    stochastic: StochasticConfig = field(default_factory=StochasticConfig)

    def __init__(
        self,
        pde: str,
        parameters: dict[str, Any] | None = None,
        domain: DomainConfig | None = None,
        inputs: dict[str, InputConfig] | None = None,
        boundary_conditions: dict[str, BoundaryFieldConfig] | None = None,
        output: OutputConfig | None = None,
        solver: SolverConfig | None = None,
        time: TimeConfig | None = None,
        seed: int | None = None,
        coefficients: dict[str, FieldExpressionConfig] | None = None,
        stochastic: StochasticConfig | None = None,
    ) -> None:
        if parameters is None:
            raise TypeError("SimulationConfig requires explicit parameters.")
        if domain is None:
            raise TypeError("SimulationConfig requires an explicit domain.")
        if inputs is None:
            raise TypeError("SimulationConfig requires explicit inputs.")
        if boundary_conditions is None:
            raise TypeError("SimulationConfig requires explicit boundary_conditions.")
        if output is None:
            raise TypeError("SimulationConfig requires explicit output settings.")
        if solver is None:
            raise TypeError("SimulationConfig requires explicit solver settings.")

        self.pde = pde
        self.parameters = parameters
        self.domain = domain
        self.inputs = inputs
        self.boundary_conditions = boundary_conditions
        self.output = output
        self.solver = solver
        self.time = time
        self.seed = seed
        self.coefficients = {} if coefficients is None else coefficients
        self.stochastic = StochasticConfig() if stochastic is None else stochastic

    @property
    def dt(self) -> float | None:
        """Compatibility accessor for time step size."""
        if self.time is None:
            return None
        return self.time.dt

    @property
    def t_end(self) -> float | None:
        """Compatibility accessor for final time."""
        if self.time is None:
            return None
        return self.time.t_end

    @property
    def output_resolution(self) -> list[int]:
        """Return the configured output grid resolution."""
        return self.output.resolution

    def input(self, name: str) -> InputConfig:
        """Return a configured input by name."""
        if name not in self.inputs:
            raise KeyError(f"Unknown input '{name}'")
        return self.inputs[name]

    def coefficient(self, name: str) -> FieldExpressionConfig:
        """Return a configured coefficient by name."""
        if name not in self.coefficients:
            raise KeyError(f"Unknown coefficient '{name}'")
        return self.coefficients[name]

    def boundary_field(self, name: str) -> BoundaryFieldConfig:
        """Return configured boundary conditions for one BC field."""
        if name not in self.boundary_conditions:
            raise KeyError(f"Unknown boundary field '{name}'")
        return self.boundary_conditions[name]

    def field(self, name: str) -> InputConfig:
        """Compatibility accessor for input configs."""
        return self.input(name)

    def output_mode(self, name: str) -> str:
        """Return the configured output mode for a named output."""
        if name not in self.output.fields:
            raise KeyError(f"Unknown output '{name}'")
        return self.output.fields[name].mode

    def stochastic_state(self, name: str) -> StateStochasticConfig | None:
        """Return the stochastic forcing configured for one state, if any."""
        return self.stochastic.states.get(name)

    def stochastic_coefficient(self, name: str) -> CoefficientStochasticConfig | None:
        """Return the stochastic overlay configured for one coefficient, if any."""
        return self.stochastic.coefficients.get(name)

    @property
    def has_stochastic(self) -> bool:
        """Return whether stochastic forcing or random media are enabled."""
        return self.stochastic.enabled

    @property
    def has_periodic_boundary_conditions(self) -> bool:
        """Return whether any BC field uses periodic side pairs."""
        return any(
            field_config.has_periodic
            for field_config in self.boundary_conditions.values()
        )


__all__ = [
    "BoundaryConditionConfig",
    "BoundaryFieldConfig",
    "CoefficientSmoothingConfig",
    "CoefficientStochasticConfig",
    "DomainConfig",
    "FieldExpressionConfig",
    "InputConfig",
    "OutputConfig",
    "OutputSelectionConfig",
    "PeriodicMapConfig",
    "SimulationConfig",
    "SolverConfig",
    "StateStochasticConfig",
    "StochasticConfig",
    "TimeConfig",
]

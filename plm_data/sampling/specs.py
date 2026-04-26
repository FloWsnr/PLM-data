"""Spec-owned random-run sampling contracts."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from plm_data.core.runtime_config import DomainConfig, FieldExpressionConfig
    from plm_data.sampling.context import SamplingContext

DomainSampler = Callable[["SamplingContext", dict[str, Any]], "DomainConfig"]
CoefficientSampler = Callable[
    ["SamplingContext", "DomainConfig", dict[str, float]],
    dict[str, "FieldExpressionConfig"],
]


@dataclass(frozen=True)
class RandomDomainProfile:
    """One random parameterization exposed by a domain spec."""

    name: str
    sample: DomainSampler
    description: str = ""


@dataclass(frozen=True)
class RandomDomainConstraint:
    """PDE-side limits for one compatible domain profile."""

    domain: str
    profile: str = "default"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RandomOutputSpec:
    """Random-run output fields and sampling limits."""

    fields: dict[str, str]
    base_resolution: tuple[int, int]
    num_frames: int
    resolution_jitter: tuple[int, int]


@dataclass(frozen=True)
class RandomTimeSpec:
    """Random-run time stepping limits for one PDE."""

    dt: float
    t_end: float


@dataclass(frozen=True)
class RandomPDEOptions:
    """Random-run options declared by one PDE spec."""

    case_name: str
    solver_strategy: str
    time: RandomTimeSpec
    output: RandomOutputSpec
    domain_constraints: tuple[RandomDomainConstraint, ...]
    coefficient_sampler: CoefficientSampler | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    allow_unconstrained_domains: bool = False

    def constraints_for(self, domain_name: str, profile_name: str) -> dict[str, Any]:
        """Return PDE-specific domain constraints for one profile, if any."""
        for constraint in self.domain_constraints:
            if constraint.domain == domain_name and constraint.profile == profile_name:
                return dict(constraint.params)
        return {}

    def allows_domain_profile(self, domain_name: str, profile_name: str) -> bool:
        """Return whether this PDE may use a domain profile."""
        if self.allow_unconstrained_domains:
            return True
        return any(
            constraint.domain == domain_name and constraint.profile == profile_name
            for constraint in self.domain_constraints
        )

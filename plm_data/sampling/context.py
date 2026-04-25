"""Sampling context objects for random runtime-config generation."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SamplingContext:
    """Shared state passed through contextual samplers."""

    seed: int
    attempt: int
    rng: np.random.Generator
    pde_name: str | None = None
    domain_name: str | None = None
    values: dict[str, Any] = field(default_factory=dict)

    def child(self, label: str) -> "SamplingContext":
        """Return a deterministic child context for one sampling stream."""
        from plm_data.sampling.values import rng_for_stream

        return SamplingContext(
            seed=self.seed,
            attempt=self.attempt,
            rng=rng_for_stream(self.seed, f"attempt.{self.attempt}.{label}"),
            pde_name=self.pde_name,
            domain_name=self.domain_name,
            values=self.values,
        )

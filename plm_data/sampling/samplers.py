"""Small contextual sampler helpers."""

from typing import Any

import numpy as np

from plm_data.core.sampling import rng_for_stream
from plm_data.sampling.context import SamplingContext


def attempt_rng(seed: int, attempt: int, stream: str) -> np.random.Generator:
    """Return a deterministic RNG stream for one seed and sampling attempt."""
    return rng_for_stream(seed, f"random.{attempt}.{stream}")


def uniform(
    context: SamplingContext,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    """Sample a float uniformly and store it in the context."""
    value = float(context.child(name).rng.uniform(minimum, maximum))
    context.values[name] = value
    return value


def randint(
    context: SamplingContext,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    """Sample an inclusive integer uniformly and store it in the context."""
    value = int(context.child(name).rng.integers(minimum, maximum + 1))
    context.values[name] = value
    return value


def choose(
    context: SamplingContext,
    name: str,
    choices: list[Any] | tuple[Any, ...],
) -> Any:
    """Choose one item uniformly and store it in the context."""
    if not choices:
        raise ValueError(f"Cannot sample '{name}' from an empty choice list.")
    index = int(context.child(name).rng.integers(0, len(choices)))
    value = choices[index]
    context.values[name] = value
    return value

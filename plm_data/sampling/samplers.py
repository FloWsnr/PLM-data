"""Small contextual sampler helpers."""

from typing import Any

import numpy as np

from plm_data.pdes.metadata import PDEParameter
from plm_data.sampling.context import SamplingContext
from plm_data.sampling.values import rng_for_stream


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


def sample_numeric_parameter(
    context: SamplingContext,
    stream: str,
    parameter: PDEParameter,
) -> float | int:
    """Sample one numeric parameter from its spec and store it in the context."""
    if parameter.default is not None:
        value: float | int = parameter.default
    elif parameter.sampler is not None:
        value = parameter.sampler(context, stream)
    elif parameter.sampling_min is not None and parameter.sampling_max is not None:
        if parameter.kind == "int":
            value = randint(
                context,
                stream,
                int(round(parameter.sampling_min)),
                int(round(parameter.sampling_max)),
            )
        else:
            value = uniform(
                context,
                stream,
                float(parameter.sampling_min),
                float(parameter.sampling_max),
            )
    else:
        raise ValueError(
            f"PDE parameter '{parameter.name}' does not declare a default, "
            "sampler, or sampling range."
        )

    if parameter.kind == "int":
        value = int(round(value))
    else:
        value = float(value)
    if parameter.sampling_min is not None and value < float(parameter.sampling_min):
        raise ValueError(
            f"sampled parameter '{stream}' must be >= sampling_min "
            f"{parameter.sampling_min}. Got {value!r}."
        )
    if parameter.sampling_max is not None and value > float(parameter.sampling_max):
        raise ValueError(
            f"sampled parameter '{stream}' must be <= sampling_max "
            f"{parameter.sampling_max}. Got {value!r}."
        )
    parameter.validate_value(value, context=f"sampled parameter '{stream}'")
    context.values[stream] = value
    return value

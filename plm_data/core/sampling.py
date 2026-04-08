"""Shared seeded sampling helpers for declarative config values."""

import hashlib
from typing import Any

import numpy as np


def is_param_ref(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("param:")


def resolve_numeric_or_param_ref(
    value: Any,
    parameters: dict[str, float],
    context: str,
) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if is_param_ref(value):
        parameter_name = value[len("param:") :]
        if parameter_name not in parameters:
            raise ValueError(
                f"{context} references unknown parameter '{parameter_name}'. "
                f"Available parameters: {sorted(parameters)}."
            )
        return float(parameters[parameter_name])
    raise ValueError(
        f"{context} must be a number or 'param:<name>' reference. Got {value!r}."
    )


def is_sampler_spec(value: Any) -> bool:
    return isinstance(value, dict) and "sample" in value


def contains_sampler_spec(value: Any) -> bool:
    if is_sampler_spec(value):
        return True
    if isinstance(value, dict):
        return any(contains_sampler_spec(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_sampler_spec(item) for item in value)
    return False


def rng_for_stream(seed: int, stream_id: str | None) -> np.random.Generator:
    if not stream_id:
        return np.random.default_rng(seed)

    digest = hashlib.sha256(f"{seed}:{stream_id}".encode("utf-8")).digest()
    entropy = [seed] + [
        int.from_bytes(digest[index : index + 4], "little") for index in range(0, 16, 4)
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def sample_number(
    value: Any,
    *,
    parameters: dict[str, float],
    rng: np.random.Generator | None,
    context: str,
) -> float:
    if not is_sampler_spec(value):
        return resolve_numeric_or_param_ref(value, parameters, context)

    if rng is None:
        raise ValueError(
            f"{context} requires an explicit seed from the config or '--seed'."
        )

    sample_type = value["sample"]
    if sample_type == "uniform":
        minimum = resolve_numeric_or_param_ref(
            value["min"], parameters, f"{context}.min"
        )
        maximum = resolve_numeric_or_param_ref(
            value["max"], parameters, f"{context}.max"
        )
        return float(rng.uniform(minimum, maximum))
    if sample_type == "normal":
        mean = resolve_numeric_or_param_ref(
            value["mean"], parameters, f"{context}.mean"
        )
        std = resolve_numeric_or_param_ref(value["std"], parameters, f"{context}.std")
        return float(rng.normal(mean, std))
    if sample_type == "randint":
        minimum = int(
            round(
                resolve_numeric_or_param_ref(
                    value["min"],
                    parameters,
                    f"{context}.min",
                )
            )
        )
        maximum = int(
            round(
                resolve_numeric_or_param_ref(
                    value["max"],
                    parameters,
                    f"{context}.max",
                )
            )
        )
        return float(rng.integers(minimum, maximum + 1))

    raise ValueError(f"{context} uses unknown sampler '{sample_type}'.")


def sample_integer(
    value: Any,
    *,
    parameters: dict[str, float],
    rng: np.random.Generator | None,
    context: str,
) -> int:
    sampled = sample_number(value, parameters=parameters, rng=rng, context=context)
    integer = int(round(sampled))
    if abs(sampled - integer) > 1.0e-9:
        raise ValueError(f"{context} expected an integer-valued sample, got {sampled}.")
    return integer


def sample_coordinate_list(
    values: Any,
    *,
    gdim: int,
    parameters: dict[str, float],
    rng: np.random.Generator | None,
    context: str,
) -> list[float]:
    if not isinstance(values, list) or len(values) != gdim:
        raise ValueError(
            f"{context} must have {gdim} entries in {gdim}D. Got {values!r}."
        )
    return [
        sample_number(
            value,
            parameters=parameters,
            rng=rng,
            context=f"{context}[{index}]",
        )
        for index, value in enumerate(values)
    ]

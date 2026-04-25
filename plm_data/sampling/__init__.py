"""Random simulation sampling interfaces."""

from typing import TYPE_CHECKING, Any

_CONTEXT_EXPORTS = {"SamplingContext"}
_RUNTIME_CONFIG_EXPORTS = {
    "DEFAULT_RANDOM_OUTPUT_FORMATS",
    "MIGRATED_RANDOM_PDES",
    "RandomPDEProfile",
    "SampledRuntimeConfig",
    "list_random_pde_cases",
    "list_random_pde_profiles",
    "sample_random_runtime_config",
    "validate_runtime_config",
}
_SAMPLER_EXPORTS = {"attempt_rng", "choose", "randint", "uniform"}
_VALUE_EXPORTS = {
    "contains_sampler_spec",
    "is_param_ref",
    "is_sampler_spec",
    "resolve_numeric_or_param_ref",
    "rng_for_stream",
    "sample_coordinate_list",
    "sample_integer",
    "sample_number",
}

if TYPE_CHECKING:
    from plm_data.sampling.context import SamplingContext as SamplingContext
    from plm_data.sampling.runtime_config import (
        DEFAULT_RANDOM_OUTPUT_FORMATS as DEFAULT_RANDOM_OUTPUT_FORMATS,
        MIGRATED_RANDOM_PDES as MIGRATED_RANDOM_PDES,
        RandomPDEProfile as RandomPDEProfile,
        SampledRuntimeConfig as SampledRuntimeConfig,
        list_random_pde_cases as list_random_pde_cases,
        list_random_pde_profiles as list_random_pde_profiles,
        sample_random_runtime_config as sample_random_runtime_config,
        validate_runtime_config as validate_runtime_config,
    )
    from plm_data.sampling.samplers import (
        attempt_rng as attempt_rng,
        choose as choose,
        randint as randint,
        uniform as uniform,
    )
    from plm_data.sampling.values import (
        contains_sampler_spec as contains_sampler_spec,
        is_param_ref as is_param_ref,
        is_sampler_spec as is_sampler_spec,
        resolve_numeric_or_param_ref as resolve_numeric_or_param_ref,
        rng_for_stream as rng_for_stream,
        sample_coordinate_list as sample_coordinate_list,
        sample_integer as sample_integer,
        sample_number as sample_number,
    )

__all__ = [
    "DEFAULT_RANDOM_OUTPUT_FORMATS",
    "MIGRATED_RANDOM_PDES",
    "RandomPDEProfile",
    "SampledRuntimeConfig",
    "SamplingContext",
    "attempt_rng",
    "choose",
    "contains_sampler_spec",
    "is_param_ref",
    "is_sampler_spec",
    "list_random_pde_cases",
    "list_random_pde_profiles",
    "randint",
    "resolve_numeric_or_param_ref",
    "rng_for_stream",
    "sample_coordinate_list",
    "sample_integer",
    "sample_number",
    "sample_random_runtime_config",
    "uniform",
    "validate_runtime_config",
]


def __getattr__(name: str) -> Any:
    if name in _CONTEXT_EXPORTS:
        from plm_data.sampling import context

        value = getattr(context, name)
    elif name in _RUNTIME_CONFIG_EXPORTS:
        from plm_data.sampling import runtime_config

        value = getattr(runtime_config, name)
    elif name in _SAMPLER_EXPORTS:
        from plm_data.sampling import samplers

        value = getattr(samplers, name)
    elif name in _VALUE_EXPORTS:
        from plm_data.sampling import values

        value = getattr(values, name)
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    globals()[name] = value
    return value

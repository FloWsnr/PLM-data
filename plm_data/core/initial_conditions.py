"""Initial condition helpers for scalar and vector fields."""

import numpy as np
from dolfinx import fem

from plm_data.core.config import FieldExpressionConfig
from plm_data.core.spatial_fields import (
    build_interpolator,
    build_vector_interpolator,
    scalar_expression_to_config,
)


def apply_ic(
    func: fem.Function,
    ic_config: FieldExpressionConfig,
    parameters: dict[str, float],
    seed: int | None = None,
) -> None:
    """Apply a scalar initial condition to a DOLFINx function in-place."""
    if ic_config.is_componentwise:
        raise ValueError("apply_ic expects a scalar initial-condition config")

    ic_type = ic_config.type
    if ic_type == "custom":
        return

    if ic_type == "random_perturbation":
        rng = np.random.default_rng(seed)
        mean = ic_config.params["mean"]
        std = ic_config.params["std"]
        func.x.array[:] = rng.normal(mean, std, size=func.x.array.shape)
        return

    interpolator = build_interpolator(
        scalar_expression_to_config(ic_config),
        parameters,
    )
    if interpolator is not None:
        func.interpolate(interpolator)


def apply_vector_ic(
    func: fem.Function,
    ic_config: FieldExpressionConfig,
    parameters: dict[str, float],
    seed: int | None = None,
) -> None:
    """Apply a vector initial condition to a DOLFINx vector function."""
    if ic_config.type == "custom" and not ic_config.is_componentwise:
        return

    if ic_config.type == "random_perturbation":
        raise ValueError("Vector initial conditions do not support random_perturbation")

    gdim = func.function_space.mesh.geometry.dim
    interpolator = build_vector_interpolator(ic_config, gdim, parameters)
    if interpolator is not None:
        func.interpolate(interpolator)

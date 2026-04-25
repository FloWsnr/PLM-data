"""Static stochastic coefficient construction."""

from typing import TYPE_CHECKING

import numpy as np
import ufl
from dolfinx import default_real_type, fem

from plm_data.core.runtime_config import (
    CoefficientSmoothingConfig,
    CoefficientStochasticConfig,
    FieldExpressionConfig,
)
from plm_data.fields.expressions import scalar_expression_to_config
from plm_data.fields.interpolation import build_interpolator
from plm_data.fields.ufl import build_ufl_field
from plm_data.stochastic.noise import _ScalarCellNoise

if TYPE_CHECKING:
    from plm_data.pdes.base import ProblemInstance


def _interpolate_scalar_expression(
    function: fem.Function,
    expression: FieldExpressionConfig,
    parameters: dict[str, float],
    *,
    context: str,
) -> None:
    interpolator = build_interpolator(
        scalar_expression_to_config(expression),
        parameters,
    )
    if interpolator is None:
        raise ValueError(f"{context} cannot use a custom scalar expression.")
    function.interpolate(interpolator)
    function.x.scatter_forward()


def _solve_linear_problem(problem, *, context: str) -> None:
    problem.solve()
    reason = problem.solver.getConvergedReason()
    if reason <= 0:
        raise RuntimeError(f"{context} did not converge (KSP reason={reason})")


def _project_scalar_noise(
    problem: "ProblemInstance",
    source: fem.Function,
    *,
    name: str,
) -> fem.Function:
    target_space = fem.functionspace(problem.msh, ("Lagrange", 1))
    target = fem.Function(target_space, name=name)
    trial = ufl.TrialFunction(target_space)
    test = ufl.TestFunction(target_space)
    projection_problem = problem.create_linear_problem(
        ufl.inner(trial, test) * ufl.dx,
        ufl.inner(source, test) * ufl.dx,
        u=target,
        bcs=[],
        petsc_options_prefix=f"plm_stochastic_{problem.spec.name}_{name}_project_",
    )
    _solve_linear_problem(
        projection_problem,
        context=f"Stochastic projection '{name}'",
    )
    return target


def _diffusion_smooth(
    problem: "ProblemInstance",
    field: fem.Function,
    *,
    name: str,
    smoothing: CoefficientSmoothingConfig,
) -> fem.Function:
    solution = fem.Function(field.function_space, name=name)
    previous = fem.Function(field.function_space, name=f"{name}_prev")
    solution.x.array[:] = field.x.array
    solution.x.scatter_forward()

    trial = ufl.TrialFunction(field.function_space)
    test = ufl.TestFunction(field.function_space)
    tau = fem.Constant(problem.msh, default_real_type(smoothing.pseudo_dt))
    smoothing_problem = problem.create_linear_problem(
        ufl.inner(trial, test) * ufl.dx
        + tau * ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx,
        ufl.inner(previous, test) * ufl.dx,
        u=solution,
        bcs=[],
        petsc_options_prefix=f"plm_stochastic_{problem.spec.name}_{name}_smooth_",
    )

    for _ in range(smoothing.steps):
        previous.x.array[:] = solution.x.array
        previous.x.scatter_forward()
        _solve_linear_problem(
            smoothing_problem,
            context=f"Stochastic smoothing '{name}'",
        )

    return solution


def _randomized_scalar_coefficient(
    problem: "ProblemInstance",
    *,
    name: str,
    base_expression: FieldExpressionConfig,
    stochastic: CoefficientStochasticConfig,
) -> fem.Function:
    stream_root = f"{problem.spec.name}.coefficient.{name}"
    overlay_sampler = _ScalarCellNoise(
        problem.msh,
        seed=problem.config.seed,
        stream_root=stream_root,
        volume_scaling=False,
    )
    overlay_sampler.fill()

    if stochastic.smoothing is None:
        coefficient_space = overlay_sampler.space
        overlay_function = overlay_sampler.function
    else:
        overlay_function = _project_scalar_noise(
            problem,
            overlay_sampler.function,
            name=f"{name}_overlay",
        )
        overlay_function = _diffusion_smooth(
            problem,
            overlay_function,
            name=f"{name}_overlay",
            smoothing=stochastic.smoothing,
        )
        coefficient_space = overlay_function.function_space

    coefficient = fem.Function(coefficient_space, name=name)
    _interpolate_scalar_expression(
        coefficient,
        base_expression,
        problem.config.parameters,
        context=f"Coefficient '{name}'",
    )

    overlay = overlay_function.x.array
    if stochastic.mode == "additive":
        coefficient.x.array[:] = coefficient.x.array + stochastic.std * overlay
    elif stochastic.mode == "multiplicative":
        coefficient.x.array[:] = coefficient.x.array * (1.0 + stochastic.std * overlay)
    else:
        raise ValueError(
            f"Unsupported stochastic coefficient mode '{stochastic.mode}'."
        )

    if stochastic.clamp_min is not None:
        coefficient.x.array[:] = np.maximum(coefficient.x.array, stochastic.clamp_min)
    coefficient.x.scatter_forward()
    return coefficient


def build_scalar_coefficient(
    problem: "ProblemInstance",
    name: str,
):
    """Build a scalar coefficient, optionally randomized as a static medium."""
    coefficient_expression = problem.config.coefficient(name)
    stochastic = problem.config.stochastic_coefficient(name)
    if stochastic is not None and stochastic.std > 0.0:
        return _randomized_scalar_coefficient(
            problem,
            name=name,
            base_expression=coefficient_expression,
            stochastic=stochastic,
        )

    return build_ufl_field(
        problem.msh,
        scalar_expression_to_config(coefficient_expression),
        problem.config.parameters,
    )

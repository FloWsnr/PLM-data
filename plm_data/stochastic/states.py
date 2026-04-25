"""State stochastic forcing terms."""

from typing import TYPE_CHECKING

import ufl

from plm_data.stochastic.noise import DynamicStateNoiseRuntime

if TYPE_CHECKING:
    from plm_data.pdes.base import ProblemInstance


def build_scalar_state_stochastic_term(
    problem: "ProblemInstance",
    *,
    state_name: str,
    previous_state,
    test,
    dt: float,
) -> tuple[ufl.Form | None, DynamicStateNoiseRuntime | None]:
    """Build a scalar stochastic forcing contribution for one state."""
    stochastic = problem.config.stochastic_state(state_name)
    if stochastic is None or stochastic.intensity == 0.0:
        return None, None

    runtime = DynamicStateNoiseRuntime(
        problem.msh,
        seed=problem.config.seed,
        stream_root=f"{problem.spec.name}.state.{state_name}",
        dt=dt,
        state_shape="scalar",
        stochastic=stochastic,
    )
    return ufl.inner(runtime.forcing_expr(previous_state), test) * ufl.dx, runtime


def build_vector_state_stochastic_term(
    problem: "ProblemInstance",
    *,
    state_name: str,
    previous_state,
    test,
    dt: float,
) -> tuple[ufl.Form | None, DynamicStateNoiseRuntime | None]:
    """Build a vector stochastic forcing contribution for one state."""
    stochastic = problem.config.stochastic_state(state_name)
    if stochastic is None or stochastic.intensity == 0.0:
        return None, None

    runtime = DynamicStateNoiseRuntime(
        problem.msh,
        seed=problem.config.seed,
        stream_root=f"{problem.spec.name}.state.{state_name}",
        dt=dt,
        state_shape="vector",
        stochastic=stochastic,
    )
    return ufl.inner(runtime.forcing_expr(previous_state), test) * ufl.dx, runtime

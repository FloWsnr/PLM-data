"""Tests for shared stochastic forcing and random-media support."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import ufl
from dolfinx import fem

from plm_data.core.runtime_config import (
    BoundaryConditionConfig,
    CoefficientSmoothingConfig,
    CoefficientStochasticConfig,
    DomainConfig,
    StateStochasticConfig,
    StochasticConfig,
    TimeConfig,
)
from plm_data.domains import create_domain
from plm_data.stochastic import build_scalar_coefficient
from plm_data.stochastic.noise import (
    DynamicStateNoiseRuntime,
    _ScalarCellNoise,
    _cell_volumes,
)
from plm_data.stochastic.states import build_vector_state_stochastic_term
from plm_data.pdes import get_pde
from tests.runtime_helpers import (
    assert_expected_output_arrays,
    assert_nontrivial,
    constant,
    make_heat_config,
    run_pde,
    scalar_expr,
)


def _homogeneous_scalar_neumann_boundary_conditions(
    *, gdim: int
) -> dict[str, BoundaryConditionConfig]:
    if gdim == 2:
        sides = ("x-", "x+", "y-", "y+")
    else:
        raise ValueError(f"Expected 2D scalar boundaries, got {gdim}D")
    return {
        side: BoundaryConditionConfig(type="neumann", value=constant(0.0))
        for side in sides
    }


def _rectangle_mesh():
    domain = create_domain(
        DomainConfig(
            type="rectangle",
            params={"size": [1.0, 1.0], "mesh_resolution": [4, 4]},
        )
    )
    return domain.mesh


def test_scalar_cell_noise_is_reproducible_and_static_without_step_index():
    mesh = _rectangle_mesh()
    sampler_a = _ScalarCellNoise(
        mesh,
        seed=17,
        stream_root="tests.scalar_noise",
        volume_scaling=False,
    )
    sampler_b = _ScalarCellNoise(
        mesh,
        seed=17,
        stream_root="tests.scalar_noise",
        volume_scaling=False,
    )

    sampler_a.fill()
    first = sampler_a.function.x.array.copy()
    sampler_a.fill()
    second = sampler_a.function.x.array.copy()
    sampler_b.fill()

    assert np.allclose(first, second)
    assert np.allclose(first, sampler_b.function.x.array)


def test_scalar_cell_noise_changes_across_timesteps():
    mesh = _rectangle_mesh()
    sampler = _ScalarCellNoise(
        mesh,
        seed=23,
        stream_root="tests.dynamic_noise",
        volume_scaling=True,
    )

    sampler.fill(0)
    step_zero = sampler.function.x.array.copy()
    sampler.fill(1)
    step_one = sampler.function.x.array.copy()

    assert not np.allclose(step_zero, step_one)


def test_scalar_cell_noise_requires_seed():
    mesh = _rectangle_mesh()
    sampler = _ScalarCellNoise(
        mesh,
        seed=None,
        stream_root="tests.seed_required",
        volume_scaling=False,
    )

    with pytest.raises(ValueError, match="explicit seed"):
        sampler.fill()


def test_dynamic_noise_volume_scaling_matches_cell_volumes():
    mesh = _rectangle_mesh()
    sampler = _ScalarCellNoise(
        mesh,
        seed=31,
        stream_root="tests.volume_scale",
        volume_scaling=True,
    )

    assert np.allclose(sampler._scale, 1.0 / np.sqrt(_cell_volumes(mesh)))


def test_dynamic_vector_noise_uses_independent_component_streams():
    mesh = _rectangle_mesh()
    runtime = DynamicStateNoiseRuntime(
        mesh,
        seed=19,
        stream_root="tests.vector_runtime",
        dt=0.05,
        state_shape="vector",
        stochastic=StateStochasticConfig(
            coupling="additive",
            intensity=1.0,
        ),
    )

    runtime.update(0)
    component_0 = runtime._components[0].function.x.array.copy()
    component_1 = runtime._components[1].function.x.array.copy()

    assert not np.allclose(component_0, component_1)


def test_build_vector_state_stochastic_term_returns_form_and_runtime():
    mesh = _rectangle_mesh()
    V = fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    previous_state = fem.Function(V)
    test_function = ufl.TestFunction(V)
    stochastic = StateStochasticConfig(coupling="additive", intensity=0.5)
    problem = SimpleNamespace(
        msh=mesh,
        config=SimpleNamespace(
            seed=29,
            stochastic_state=lambda name: stochastic if name == "velocity" else None,
        ),
        spec=SimpleNamespace(name="unit_test"),
    )

    form, runtime = build_vector_state_stochastic_term(
        problem,
        state_name="velocity",
        previous_state=previous_state,
        test=test_function,
        dt=0.05,
    )

    assert form is not None
    assert isinstance(runtime, DynamicStateNoiseRuntime)
    assert runtime.state_shape == "vector"


def test_build_scalar_coefficient_randomization_is_reproducible_and_clamped(
    heat_config,
):
    config = replace(
        heat_config,
        stochastic=StochasticConfig(
            coefficients={
                "kappa": CoefficientStochasticConfig(
                    mode="multiplicative",
                    std=0.4,
                    smoothing=CoefficientSmoothingConfig(
                        pseudo_dt=0.01,
                        steps=1,
                    ),
                    clamp_min=0.005,
                )
            }
        ),
    )
    problem = get_pde("heat").build_problem(config)
    domain_geom = problem.load_domain_geometry()
    problem.msh = domain_geom.mesh

    coefficient_a = build_scalar_coefficient(problem, "kappa")
    coefficient_b = build_scalar_coefficient(problem, "kappa")

    assert isinstance(coefficient_a, fem.Function)
    assert np.allclose(coefficient_a.x.array, coefficient_b.x.array)
    assert float(np.min(coefficient_a.x.array)) >= 0.005 - 1.0e-12
    assert float(np.std(coefficient_a.x.array)) > 0.0


def test_heat_stochastic_state_and_coefficient_run_stays_finite(tmp_path):
    config = make_heat_config(
        tmp_path,
        coefficients={"kappa": constant(0.04)},
        boundary_conditions=_homogeneous_scalar_neumann_boundary_conditions(gdim=2),
        source=scalar_expr("none"),
        initial_condition=scalar_expr(
            "gaussian_bump",
            amplitude=1.0,
            sigma=0.12,
            center=[0.45, 0.55],
        ),
        mesh_resolution=(16, 16),
        output_resolution=(8, 8),
        time=TimeConfig(dt=0.01, t_end=0.01),
    )
    config = replace(
        config,
        stochastic=StochasticConfig(
            states={"u": StateStochasticConfig(coupling="additive", intensity=0.08)},
            coefficients={
                "kappa": CoefficientStochasticConfig(
                    mode="multiplicative",
                    std=0.25,
                    smoothing=CoefficientSmoothingConfig(
                        pseudo_dt=0.01,
                        steps=1,
                    ),
                    clamp_min=0.01,
                )
            },
        ),
    )

    result, output_dir = run_pde(config)
    arrays = assert_expected_output_arrays(config, output_dir)

    assert result.solver_converged is True
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["u"][-1] - arrays["u"][0], threshold=1.0e-6)

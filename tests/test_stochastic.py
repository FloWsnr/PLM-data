"""Tests for shared stochastic forcing and random-media support."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from dolfinx import fem

from plm_data.core.config import (
    BoundaryConditionConfig,
    CoefficientSmoothingConfig,
    CoefficientStochasticConfig,
    DomainConfig,
    StateStochasticConfig,
    StochasticConfig,
    TimeConfig,
    load_config,
)
from plm_data.core.mesh import create_domain
from plm_data.core.stochastic import (
    DynamicStateNoiseRuntime,
    _ScalarCellNoise,
    _cell_volumes,
    build_scalar_coefficient,
)
from plm_data.presets import get_preset
from tests.preset_matrix import (
    assert_expected_output_arrays,
    assert_nontrivial,
    constant,
    make_advection_config,
    make_burgers_config,
    make_gray_scott_config,
    make_scalar_preset_config,
    run_preset,
    scalar_expr,
    vector_expr,
    vector_zero,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_config_dict(relative_path: str) -> dict[str, object]:
    return yaml.safe_load((_REPO_ROOT / relative_path).read_text())


def _write_yaml(tmp_path, name: str, data: dict[str, object]) -> Path:
    path = tmp_path / name
    path.write_text(yaml.dump(data))
    return path


def _homogeneous_scalar_neumann_boundary_conditions(
    *, gdim: int
) -> dict[str, BoundaryConditionConfig]:
    if gdim == 2:
        sides = ("x-", "x+", "y-", "y+")
    elif gdim == 3:
        sides = ("x-", "x+", "y-", "y+", "z-", "z+")
    else:
        raise ValueError(f"Expected 2D or 3D scalar boundaries, got {gdim}D")
    return {
        side: BoundaryConditionConfig(type="neumann", value=constant(0.0))
        for side in sides
    }


def _homogeneous_vector_dirichlet_boundary_conditions(
    *, gdim: int
) -> dict[str, BoundaryConditionConfig]:
    if gdim == 2:
        sides = ("x-", "x+", "y-", "y+")
    elif gdim == 3:
        sides = ("x-", "x+", "y-", "y+", "z-", "z+")
    else:
        raise ValueError(f"Expected 2D or 3D vector boundaries, got {gdim}D")
    return {
        side: BoundaryConditionConfig(type="dirichlet", value=vector_zero())
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


def _box_mesh():
    domain = create_domain(
        DomainConfig(
            type="box",
            params={"size": [1.0, 1.0, 1.0], "mesh_resolution": [3, 3, 3]},
        )
    )
    return domain.mesh


def test_load_config_parses_stochastic_state_and_coefficient_sections(tmp_path):
    data = _load_config_dict("configs/basic/heat/2d_single_blob_diffusion.yaml")
    data["stochastic"] = {
        "states": {"u": {"coupling": "additive", "intensity": 0.2}},
        "coefficients": {
            "kappa": {
                "mode": "multiplicative",
                "std": 0.15,
                "smoothing": {"pseudo_dt": 0.02, "steps": 2},
                "clamp_min": 0.005,
            }
        },
    }

    cfg = load_config(_write_yaml(tmp_path, "heat_stochastic.yaml", data))

    assert cfg.has_stochastic is True
    state = cfg.stochastic_state("u")
    assert state is not None
    assert state.coupling == "additive"
    assert state.intensity == pytest.approx(0.2)

    coefficient = cfg.stochastic_coefficient("kappa")
    assert coefficient is not None
    assert coefficient.mode == "multiplicative"
    assert coefficient.std == pytest.approx(0.15)
    assert coefficient.clamp_min == pytest.approx(0.005)
    assert coefficient.smoothing is not None
    assert coefficient.smoothing.pseudo_dt == pytest.approx(0.02)
    assert coefficient.smoothing.steps == 2


def test_load_config_rejects_missing_saturating_offset(tmp_path):
    data = _load_config_dict("configs/physics/gray_scott/2d_drifting_spot_stripe_patterns.yaml")
    data["stochastic"] = {
        "states": {"u": {"coupling": "saturating_self", "intensity": 0.05}}
    }

    path = _write_yaml(tmp_path, "gray_scott_missing_offset.yaml", data)
    with pytest.raises(ValueError, match="requires 'offset'"):
        load_config(path)


def test_load_config_rejects_vector_coefficient_randomization(tmp_path):
    data = _load_config_dict("configs/physics/gray_scott/2d_drifting_spot_stripe_patterns.yaml")
    data["stochastic"] = {
        "coefficients": {"velocity": {"mode": "additive", "std": 0.1}}
    }

    path = _write_yaml(tmp_path, "gray_scott_vector_coefficient.yaml", data)
    with pytest.raises(ValueError, match="does not support stochastic randomization"):
        load_config(path)


def test_load_config_rejects_stochastic_on_unsupported_preset(tmp_path):
    data = _load_config_dict("configs/basic/wave/2d_localized_pulse_propagation.yaml")
    data["stochastic"] = {"states": {"u": {"coupling": "additive", "intensity": 0.1}}}

    path = _write_yaml(tmp_path, "wave_stochastic.yaml", data)
    with pytest.raises(ValueError, match="does not support stochastic forcing"):
        load_config(path)


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


@pytest.mark.parametrize("mesh_factory", [_rectangle_mesh, _box_mesh])
def test_dynamic_noise_volume_scaling_matches_cell_volumes(mesh_factory):
    mesh = mesh_factory()
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
    problem = get_preset("heat").build_problem(config)
    domain_geom = problem.load_domain_geometry()
    problem.msh = domain_geom.mesh

    coefficient_a = build_scalar_coefficient(problem, "kappa")
    coefficient_b = build_scalar_coefficient(problem, "kappa")

    assert isinstance(coefficient_a, fem.Function)
    assert np.allclose(coefficient_a.x.array, coefficient_b.x.array)
    assert float(np.min(coefficient_a.x.array)) >= 0.005 - 1.0e-12
    assert float(np.std(coefficient_a.x.array)) > 0.0


def test_heat_stochastic_state_and_coefficient_run_stays_finite(tmp_path):
    config = make_scalar_preset_config(
        tmp_path,
        preset="heat",
        parameters={},
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

    result, output_dir = run_preset(config)
    arrays = assert_expected_output_arrays(config, output_dir)

    assert result.solver_converged is True
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["u"][-1] - arrays["u"][0], threshold=1.0e-6)


def test_gray_scott_saturating_noise_run_stays_finite(tmp_path):
    config = make_gray_scott_config(
        tmp_path,
        gdim=2,
        velocity=vector_expr(x=constant(0.0), y=constant(0.0)),
        u_initial_condition=scalar_expr(
            "gaussian_blobs",
            background=1.0,
            generators=[
                {
                    "count": 1,
                    "amplitude": -0.35,
                    "sigma": 0.12,
                    "center": [0.5, 0.5],
                }
            ],
        ),
        v_initial_condition=scalar_expr(
            "gaussian_blobs",
            background=0.0,
            generators=[
                {
                    "count": 1,
                    "amplitude": 0.22,
                    "sigma": 0.1,
                    "center": [0.5, 0.5],
                }
            ],
        ),
        u_boundary_conditions=_homogeneous_scalar_neumann_boundary_conditions(gdim=2),
        v_boundary_conditions=_homogeneous_scalar_neumann_boundary_conditions(gdim=2),
        mesh_resolution=(18, 18),
        output_resolution=(8, 8),
        time=TimeConfig(dt=1.0, t_end=1.0),
    )
    config = replace(
        config,
        stochastic=StochasticConfig(
            states={
                "u": StateStochasticConfig(
                    coupling="saturating_self",
                    intensity=0.05,
                    offset=0.2,
                ),
                "v": StateStochasticConfig(
                    coupling="saturating_self",
                    intensity=0.05,
                    offset=0.2,
                ),
            }
        ),
    )

    result, output_dir = run_preset(config)
    arrays = assert_expected_output_arrays(config, output_dir)

    assert result.solver_converged is True
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["u"][-1] - arrays["u"][0], threshold=1.0e-6)
    assert_nontrivial(arrays["v"][-1] - arrays["v"][0], threshold=1.0e-6)


def test_burgers_vector_noise_run_stays_finite(tmp_path):
    config = make_burgers_config(
        tmp_path,
        gdim=2,
        parameters={"nu": 0.02},
        boundary_conditions=_homogeneous_vector_dirichlet_boundary_conditions(gdim=2),
        source=scalar_expr("none"),
        initial_condition=vector_expr(
            x=scalar_expr(
                "gaussian_bump",
                amplitude=0.7,
                sigma=0.14,
                center=[0.42, 0.5],
            ),
            y=scalar_expr("sine_product", amplitude=0.15, kx=1.0, ky=1.0),
        ),
        mesh_resolution=(14, 14),
        output_resolution=(8, 8),
        time=TimeConfig(dt=0.02, t_end=0.02),
    )
    config = replace(
        config,
        stochastic=StochasticConfig(
            states={
                "velocity": StateStochasticConfig(
                    coupling="additive",
                    intensity=0.04,
                )
            }
        ),
    )

    result, output_dir = run_preset(config)
    arrays = assert_expected_output_arrays(config, output_dir)

    assert result.solver_converged is True
    assert result.num_timesteps == 1
    assert_nontrivial(
        arrays["velocity_x"][-1] - arrays["velocity_x"][0],
        threshold=1.0e-6,
    )
    assert_nontrivial(
        arrays["velocity_y"][-1] - arrays["velocity_y"][0],
        threshold=1.0e-6,
    )


def test_advection_3d_stochastic_run_stays_finite(tmp_path):
    config = make_advection_config(
        tmp_path,
        gdim=3,
        velocity=vector_expr(
            x=constant(0.1),
            y=constant(-0.05),
            z=constant(0.04),
        ),
        diffusivity=constant(0.01),
        boundary_conditions=_homogeneous_scalar_neumann_boundary_conditions(gdim=3),
        source=scalar_expr("none"),
        initial_condition=scalar_expr(
            "gaussian_bump",
            amplitude=1.0,
            sigma=0.12,
            center=[0.35, 0.5, 0.55],
        ),
        mesh_resolution=(6, 6, 5),
        output_resolution=(4, 4, 4),
        time=TimeConfig(dt=0.01, t_end=0.01),
    )
    config = replace(
        config,
        stochastic=StochasticConfig(
            states={"u": StateStochasticConfig(coupling="additive", intensity=0.06)},
            coefficients={
                "diffusivity": CoefficientStochasticConfig(
                    mode="multiplicative",
                    std=0.2,
                    clamp_min=0.002,
                )
            },
        ),
    )

    result, output_dir = run_preset(config)
    arrays = assert_expected_output_arrays(config, output_dir)

    assert result.solver_converged is True
    assert result.num_timesteps == 1
    assert_nontrivial(arrays["u"][-1] - arrays["u"][0], threshold=1.0e-6)

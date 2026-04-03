"""Tests for plm_data.core.initial_conditions."""

from typing import cast

import numpy as np
import pytest
from dolfinx import fem

from plm_data.core.config import FieldExpressionConfig
from plm_data.core.initial_conditions import apply_ic, apply_vector_ic
from plm_data.core.mesh import create_domain


def _make_function(rectangle_domain):
    domain_geom = create_domain(rectangle_domain)
    V = fem.functionspace(domain_geom.mesh, ("Lagrange", 1))
    return cast(fem.Function, fem.Function(V))


def _make_vector_function(rectangle_domain):
    domain_geom = create_domain(rectangle_domain)
    V = fem.functionspace(domain_geom.mesh, ("Lagrange", 1, (2,)))
    return cast(fem.Function, fem.Function(V))


def test_apply_gaussian_bump(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(
        type="gaussian_bump",
        params={"sigma": 0.1, "amplitude": 1.0, "center": [0.5, 0.5]},
    )
    apply_ic(f, ic, {})
    assert np.max(f.x.array) > 0.5
    assert np.min(f.x.array) >= 0.0


def test_apply_constant(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(type="constant", params={"value": 3.14})
    apply_ic(f, ic, {})
    np.testing.assert_allclose(f.x.array, 3.14)


def test_apply_gaussian_noise(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(
        type="gaussian_noise",
        params={"mean": 0.5, "std": 0.01},
    )
    apply_ic(f, ic, {}, seed=42)
    assert np.mean(f.x.array) == pytest.approx(0.5, abs=0.05)
    assert np.std(f.x.array) > 0


def test_apply_sine_product(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(
        type="sine_product",
        params={"amplitude": 1.0, "kx": 1.0, "ky": 1.0},
    )
    apply_ic(f, ic, {})
    assert np.max(f.x.array) > 0


def test_apply_gaussian_blobs(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(
        type="gaussian_blobs",
        params={
            "background": 0.2,
            "generators": [
                {
                    "count": 1,
                    "amplitude": 1.0,
                    "sigma": 0.08,
                    "center": [0.3, 0.4],
                },
                {
                    "count": 1,
                    "amplitude": -0.4,
                    "sigma": 0.05,
                    "center": [0.7, 0.6],
                },
            ],
        },
    )
    apply_ic(f, ic, {})
    assert np.max(f.x.array) > 0.8
    assert np.min(f.x.array) < 0.2


def test_apply_gaussian_blobs_generator_count_is_reproducible(rectangle_domain):
    f1 = _make_function(rectangle_domain)
    f2 = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(
        type="gaussian_blobs",
        params={
            "background": 0.0,
            "generators": [
                {
                    "count": 2,
                    "amplitude": {"sample": "uniform", "min": 0.8, "max": 1.2},
                    "sigma": {"sample": "uniform", "min": 0.05, "max": 0.09},
                    "center": [
                        {"sample": "uniform", "min": 0.2, "max": 0.8},
                        {"sample": "uniform", "min": 0.2, "max": 0.8},
                    ],
                }
            ],
        },
    )
    apply_ic(f1, ic, {}, seed=42, stream_id="u")
    apply_ic(f2, ic, {}, seed=42, stream_id="u")
    np.testing.assert_allclose(f1.x.array, f2.x.array)
    assert np.max(f1.x.array) > 0.2


def test_apply_gaussian_wave_packet(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(
        type="gaussian_wave_packet",
        params={
            "amplitude": 1.0,
            "sigma": 0.08,
            "center": [0.5, 0.5],
            "wavevector": [16.0, 0.0],
            "phase": 0.0,
        },
    )
    apply_ic(f, ic, {})

    coords = f.function_space.tabulate_dof_coordinates()
    near = np.linalg.norm(coords[:, :2] - np.array([0.5, 0.5]), axis=1) < 0.18
    far = np.linalg.norm(coords[:, :2] - np.array([0.5, 0.5]), axis=1) > 0.35
    assert np.mean(np.abs(f.x.array[near])) > 10.0 * np.mean(np.abs(f.x.array[far]))
    assert np.max(f.x.array) > 0.2
    assert np.min(f.x.array) < -0.05


def test_apply_sine_waves(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(
        type="sine_waves",
        params={
            "background": 0.1,
            "modes": [
                {"amplitude": 0.3, "cycles": [1, 2], "phase": 0.0},
                {"amplitude": -0.15, "cycles": [2, 1], "phase": 0.5},
            ],
        },
    )
    apply_ic(f, ic, {})
    assert np.max(f.x.array) > 0.1
    assert np.min(f.x.array) < 0.1


def test_apply_quadrants(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(
        type="quadrants",
        params={
            "split": [0.5, 0.5],
            "region_values": {
                "00": 0.0,
                "01": 1.0,
                "10": 2.0,
                "11": 3.0,
            },
        },
    )
    apply_ic(f, ic, {})
    assert set(np.unique(np.round(f.x.array, decimals=6))) == {0.0, 1.0, 2.0, 3.0}


def test_apply_step(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(
        type="step",
        params={"value_left": 1.0, "value_right": 0.0, "x_split": 0.5, "axis": 0},
    )
    apply_ic(f, ic, {})
    assert np.max(f.x.array) == pytest.approx(1.0)
    assert np.min(f.x.array) == pytest.approx(0.0)


def test_apply_ic_with_param_refs(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(type="constant", params={"value": "param:initial_value"})
    apply_ic(f, ic, {"initial_value": 2.71})
    np.testing.assert_allclose(f.x.array, 2.71)


def test_apply_ic_with_sampled_params_is_reproducible(rectangle_domain):
    f1 = _make_function(rectangle_domain)
    f2 = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(
        type="constant",
        params={
            "value": {
                "sample": "uniform",
                "min": -1.0,
                "max": 1.0,
            }
        },
    )
    apply_ic(f1, ic, {}, seed=42, stream_id="u")
    apply_ic(f2, ic, {}, seed=42, stream_id="u")
    np.testing.assert_allclose(f1.x.array, f2.x.array)


def test_apply_vector_ic_componentwise_gaussian_noise(rectangle_domain):
    f = _make_vector_function(rectangle_domain)
    ic = FieldExpressionConfig(
        components={
            "x": FieldExpressionConfig(
                type="gaussian_noise",
                params={"mean": 0.0, "std": 0.1},
            ),
            "y": FieldExpressionConfig(
                type="gaussian_noise",
                params={"mean": 0.0, "std": 0.1},
            ),
        }
    )
    apply_vector_ic(f, ic, {}, seed=42, stream_id="velocity")
    block_size = f.function_space.dofmap.index_map_bs
    x_values = np.copy(f.x.array[0::block_size])
    y_values = np.copy(f.x.array[1::block_size])
    assert np.std(x_values) > 0.0
    assert np.std(y_values) > 0.0
    assert not np.allclose(x_values, y_values)


def test_custom_ic_is_noop(rectangle_domain):
    f = _make_function(rectangle_domain)
    f.x.array[:] = 99.0
    ic = FieldExpressionConfig(type="custom", params={})
    apply_ic(f, ic, {})
    np.testing.assert_allclose(f.x.array, 99.0)


def test_unknown_ic_raises(rectangle_domain):
    f = _make_function(rectangle_domain)
    ic = FieldExpressionConfig(type="nonexistent", params={})
    with pytest.raises(ValueError, match="Unknown initial-condition type"):
        apply_ic(f, ic, {})

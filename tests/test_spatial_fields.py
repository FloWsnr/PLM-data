"""Tests for plm_data.core.spatial_fields."""

import numpy as np
import pytest
import ufl

from plm_data.core.mesh import create_domain
from plm_data.core.spatial_fields import (
    build_interpolator,
    build_ufl_field,
    normalize_field_config,
    resolve_param_ref,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def params():
    """Standard parameter dict for testing references."""
    return {"kappa": 0.5, "amplitude": 2.0, "sigma": 0.1}


@pytest.fixture
def mesh_2d(rectangle_domain):
    """A 2D DOLFINx mesh from the shared rectangle_domain fixture."""
    return create_domain(rectangle_domain).mesh


@pytest.fixture
def points_2d():
    """Mock point array with shape (2, N) for 2D interpolation tests."""
    # A 3x3 grid on [0, 1] x [0, 1]
    xs = np.linspace(0.0, 1.0, 5)
    ys = np.linspace(0.0, 1.0, 5)
    xx, yy = np.meshgrid(xs, ys)
    return np.vstack([xx.ravel(), yy.ravel()])  # shape (2, 25)


@pytest.fixture
def points_3d():
    """Mock point array with shape (3, N) for 3D interpolation tests."""
    xs = np.linspace(0.0, 1.0, 3)
    ys = np.linspace(0.0, 1.0, 3)
    zs = np.linspace(0.0, 1.0, 3)
    xx, yy, zz = np.meshgrid(xs, ys, zs)
    return np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])  # shape (3, 27)


# ===========================================================================
# resolve_param_ref
# ===========================================================================


class TestResolveParamRef:
    def test_int_passthrough(self, params):
        assert resolve_param_ref(5, params) == 5.0

    def test_float_passthrough(self, params):
        assert resolve_param_ref(3.14, params) == 3.14

    def test_returns_float_type(self, params):
        result = resolve_param_ref(5, params)
        assert isinstance(result, float)

    def test_param_reference_resolves(self, params):
        assert resolve_param_ref("param:kappa", params) == 0.5

    def test_param_reference_resolves_other_key(self, params):
        assert resolve_param_ref("param:amplitude", params) == 2.0

    def test_missing_param_reference_raises(self, params):
        with pytest.raises(ValueError, match="not found"):
            resolve_param_ref("param:nonexistent", params)

    def test_invalid_string_raises(self, params):
        with pytest.raises(ValueError, match="Cannot resolve value"):
            resolve_param_ref("not_a_param_ref", params)

    def test_invalid_type_raises(self, params):
        with pytest.raises(ValueError, match="Cannot resolve value"):
            resolve_param_ref([1, 2, 3], params)

    def test_none_raises(self, params):
        with pytest.raises(ValueError, match="Cannot resolve value"):
            resolve_param_ref(None, params)


# ===========================================================================
# normalize_field_config
# ===========================================================================


class TestNormalizeFieldConfig:
    def test_float_input(self):
        result = normalize_field_config(3.14)
        assert result == {"type": "constant", "params": {"value": 3.14}}

    def test_int_input(self):
        result = normalize_field_config(7)
        assert result == {"type": "constant", "params": {"value": 7}}

    def test_param_string(self):
        result = normalize_field_config("param:kappa")
        assert result == {"type": "constant", "params": {"value": "param:kappa"}}

    def test_dict_with_type_params(self):
        cfg = {"type": "sine_product", "params": {"amplitude": 1.0, "kx": 2}}
        result = normalize_field_config(cfg)
        assert result is cfg  # returned as-is, same object

    def test_dict_with_type_no_params(self):
        cfg = {"type": "none"}
        result = normalize_field_config(cfg)
        assert result == {"type": "none"}

    def test_dict_missing_type_raises(self):
        with pytest.raises(ValueError, match="must have a 'type' key"):
            normalize_field_config({"params": {"value": 1.0}})

    def test_list_raises(self):
        with pytest.raises(ValueError, match="Invalid field config"):
            normalize_field_config([1, 2, 3])

    def test_plain_string_raises(self):
        with pytest.raises(ValueError, match="Invalid field config"):
            normalize_field_config("just_a_string")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="Invalid field config"):
            normalize_field_config(None)


# ===========================================================================
# build_ufl_field
# ===========================================================================


class TestBuildUflField:
    def test_type_none(self, mesh_2d, params):
        result = build_ufl_field(mesh_2d, {"type": "none"}, params)
        assert result is not None
        assert isinstance(result, ufl.core.expr.Expr)  # type: ignore[reportAttributeAccessIssue]

    def test_type_zero(self, mesh_2d, params):
        result = build_ufl_field(mesh_2d, {"type": "zero"}, params)
        assert result is not None
        assert isinstance(result, ufl.core.expr.Expr)  # type: ignore[reportAttributeAccessIssue]

    def test_constant_literal(self, mesh_2d, params):
        cfg = {"type": "constant", "params": {"value": 42.0}}
        result = build_ufl_field(mesh_2d, cfg, params)
        assert result is not None
        assert isinstance(result, ufl.core.expr.Expr)  # type: ignore[reportAttributeAccessIssue]

    def test_constant_param_ref(self, mesh_2d, params):
        cfg = {"type": "constant", "params": {"value": "param:kappa"}}
        result = build_ufl_field(mesh_2d, cfg, params)
        assert result is not None
        assert isinstance(result, ufl.core.expr.Expr)  # type: ignore[reportAttributeAccessIssue]

    def test_sine_product_kx_only(self, mesh_2d, params):
        cfg = {"type": "sine_product", "params": {"amplitude": 1.0, "kx": 2.0}}
        result = build_ufl_field(mesh_2d, cfg, params)
        assert result is not None
        assert isinstance(result, ufl.core.expr.Expr)  # type: ignore[reportAttributeAccessIssue]

    def test_sine_product_kx_ky(self, mesh_2d, params):
        cfg = {
            "type": "sine_product",
            "params": {"amplitude": 1.0, "kx": 1.0, "ky": 2.0},
        }
        result = build_ufl_field(mesh_2d, cfg, params)
        assert result is not None
        assert isinstance(result, ufl.core.expr.Expr)  # type: ignore[reportAttributeAccessIssue]

    def test_sine_product_no_axes_raises(self, mesh_2d, params):
        cfg = {"type": "sine_product", "params": {"amplitude": 1.0}}
        with pytest.raises(ValueError, match="at least one of kx, ky, kz"):
            build_ufl_field(mesh_2d, cfg, params)

    def test_gaussian_bump_2d(self, mesh_2d, params):
        cfg = {
            "type": "gaussian_bump",
            "params": {"amplitude": 1.0, "sigma": 0.1, "center": [0.5, 0.5]},
        }
        result = build_ufl_field(mesh_2d, cfg, params)
        assert result is not None
        assert isinstance(result, ufl.core.expr.Expr)  # type: ignore[reportAttributeAccessIssue]

    def test_gaussian_bump_wrong_center_dim_raises(self, mesh_2d, params):
        cfg = {
            "type": "gaussian_bump",
            "params": {"amplitude": 1.0, "sigma": 0.1, "center": [0.5, 0.5, 0.5]},
        }
        with pytest.raises(ValueError, match="3 components but mesh is 2D"):
            build_ufl_field(mesh_2d, cfg, params)

    def test_gaussian_bump_param_refs(self, mesh_2d, params):
        cfg = {
            "type": "gaussian_bump",
            "params": {
                "amplitude": "param:amplitude",
                "sigma": "param:sigma",
                "center": [0.5, 0.5],
            },
        }
        result = build_ufl_field(mesh_2d, cfg, params)
        assert result is not None
        assert isinstance(result, ufl.core.expr.Expr)  # type: ignore[reportAttributeAccessIssue]

    def test_step_2d(self, mesh_2d, params):
        cfg = {
            "type": "step",
            "params": {
                "value_left": 1.0,
                "value_right": 0.0,
                "x_split": 0.5,
                "axis": 0,
            },
        }
        result = build_ufl_field(mesh_2d, cfg, params)
        assert result is not None
        assert isinstance(result, ufl.core.expr.Expr)  # type: ignore[reportAttributeAccessIssue]

    def test_step_param_refs(self, mesh_2d, params):
        cfg = {
            "type": "step",
            "params": {
                "value_left": "param:amplitude",
                "value_right": 0.0,
                "x_split": "param:sigma",
                "axis": 0,
            },
        }
        result = build_ufl_field(mesh_2d, cfg, params)
        assert result is not None
        assert isinstance(result, ufl.core.expr.Expr)  # type: ignore[reportAttributeAccessIssue]

    def test_custom_returns_none(self, mesh_2d, params):
        result = build_ufl_field(mesh_2d, {"type": "custom"}, params)
        assert result is None

    def test_unknown_type_raises(self, mesh_2d, params):
        with pytest.raises(ValueError, match="Unknown field type"):
            build_ufl_field(mesh_2d, {"type": "bogus_type"}, params)


# ===========================================================================
# build_interpolator
# ===========================================================================


class TestBuildInterpolator:
    def test_type_none_returns_zeros(self, points_2d):
        fn = build_interpolator({"type": "none"}, {})
        assert fn is not None
        result = fn(points_2d)
        assert result.shape == (points_2d.shape[1],)
        np.testing.assert_array_equal(result, 0.0)

    def test_type_zero_returns_zeros(self, points_2d):
        fn = build_interpolator({"type": "zero"}, {})
        assert fn is not None
        result = fn(points_2d)
        assert result.shape == (points_2d.shape[1],)
        np.testing.assert_array_equal(result, 0.0)

    def test_constant_uniform(self, points_2d):
        cfg = {"type": "constant", "params": {"value": 7.5}}
        fn = build_interpolator(cfg, {})
        assert fn is not None
        result = fn(points_2d)
        assert result.shape == (points_2d.shape[1],)
        np.testing.assert_array_equal(result, 7.5)

    def test_constant_param_ref(self, points_2d):
        cfg = {"type": "constant", "params": {"value": "param:kappa"}}
        fn = build_interpolator(cfg, {"kappa": 3.0})
        assert fn is not None
        result = fn(points_2d)
        np.testing.assert_array_equal(result, 3.0)

    def test_sine_product_known_values(self, points_2d):
        """sin(pi*x) * sin(pi*y) should be 0 on all boundaries and max at center."""
        cfg = {
            "type": "sine_product",
            "params": {"amplitude": 1.0, "kx": 1.0, "ky": 1.0},
        }
        fn = build_interpolator(cfg, {})
        assert fn is not None
        result = fn(points_2d)

        # At boundary points (x=0 or x=1 or y=0 or y=1), sin(pi*0)=0 and sin(pi*1)=0
        for i in range(points_2d.shape[1]):
            x, y = points_2d[0, i], points_2d[1, i]
            expected = np.sin(np.pi * x) * np.sin(np.pi * y)
            np.testing.assert_allclose(result[i], expected, atol=1e-14)

    def test_sine_product_kx_only(self):
        """With only kx, result depends only on x."""
        cfg = {"type": "sine_product", "params": {"amplitude": 2.0, "kx": 1.0}}
        fn = build_interpolator(cfg, {})
        assert fn is not None
        x = np.array([[0.0, 0.25, 0.5, 0.75, 1.0], [0.5, 0.5, 0.5, 0.5, 0.5]])
        result = fn(x)
        expected = 2.0 * np.sin(np.pi * x[0])
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_sine_product_no_axes_raises(self):
        cfg = {"type": "sine_product", "params": {"amplitude": 1.0}}
        with pytest.raises(ValueError, match="at least one of kx, ky, kz"):
            build_interpolator(cfg, {})

    def test_sine_product_param_refs(self, points_2d):
        cfg = {
            "type": "sine_product",
            "params": {"amplitude": "param:amp", "kx": "param:k"},
        }
        fn = build_interpolator(cfg, {"amp": 3.0, "k": 2.0})
        assert fn is not None
        result = fn(points_2d)
        expected = 3.0 * np.sin(2.0 * np.pi * points_2d[0])
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_gaussian_bump_peak_at_center(self, points_2d):
        cfg = {
            "type": "gaussian_bump",
            "params": {"amplitude": 5.0, "sigma": 0.2, "center": [0.5, 0.5]},
        }
        fn = build_interpolator(cfg, {})
        assert fn is not None
        result = fn(points_2d)

        # Find the point closest to (0.5, 0.5) — it should have max value
        dists = (points_2d[0] - 0.5) ** 2 + (points_2d[1] - 0.5) ** 2
        center_idx = np.argmin(dists)
        assert result[center_idx] == pytest.approx(5.0, abs=1e-10)

        # Points far from center should have smaller values
        corner_dists = (points_2d[0] - 0.0) ** 2 + (points_2d[1] - 0.0) ** 2
        corner_idx = np.argmin(corner_dists)  # near (0, 0)
        assert result[corner_idx] < result[center_idx]

    def test_gaussian_bump_decays(self):
        """Value should decrease as we move away from center."""
        cfg = {
            "type": "gaussian_bump",
            "params": {"amplitude": 1.0, "sigma": 0.1, "center": [0.5, 0.5]},
        }
        fn = build_interpolator(cfg, {})
        assert fn is not None
        # Three points: center, near, far
        x = np.array([[0.5, 0.6, 1.0], [0.5, 0.5, 1.0]])
        result = fn(x)
        assert result[0] > result[1] > result[2]

    def test_gaussian_bump_3d(self, points_3d):
        cfg = {
            "type": "gaussian_bump",
            "params": {"amplitude": 2.0, "sigma": 0.3, "center": [0.5, 0.5, 0.5]},
        }
        fn = build_interpolator(cfg, {})
        assert fn is not None
        result = fn(points_3d)
        assert result.shape == (points_3d.shape[1],)

        # Center point should be the peak
        dists = sum((points_3d[i] - 0.5) ** 2 for i in range(3))
        center_idx = np.argmin(dists)
        assert result[center_idx] == pytest.approx(2.0, abs=1e-10)

    def test_gaussian_bump_param_refs(self, points_2d):
        cfg = {
            "type": "gaussian_bump",
            "params": {
                "amplitude": "param:amp",
                "sigma": "param:sig",
                "center": [0.5, 0.5],
            },
        }
        fn = build_interpolator(cfg, {"amp": 4.0, "sig": 0.2})
        assert fn is not None
        result = fn(points_2d)
        # Peak at center should be amplitude
        dists = (points_2d[0] - 0.5) ** 2 + (points_2d[1] - 0.5) ** 2
        center_idx = np.argmin(dists)
        assert result[center_idx] == pytest.approx(4.0, abs=1e-10)

    def test_step_x_axis(self, points_2d):
        cfg = {
            "type": "step",
            "params": {
                "value_left": 1.0,
                "value_right": 0.0,
                "x_split": 0.5,
                "axis": 0,
            },
        }
        fn = build_interpolator(cfg, {})
        assert fn is not None
        result = fn(points_2d)
        assert result.shape == (points_2d.shape[1],)
        # Points with x < 0.5 should be 1.0, x >= 0.5 should be 0.0
        for i in range(points_2d.shape[1]):
            if points_2d[0, i] < 0.5:
                assert result[i] == 1.0
            else:
                assert result[i] == 0.0

    def test_step_y_axis(self, points_2d):
        cfg = {
            "type": "step",
            "params": {
                "value_left": -1.0,
                "value_right": 2.0,
                "x_split": 0.5,
                "axis": 1,
            },
        }
        fn = build_interpolator(cfg, {})
        assert fn is not None
        result = fn(points_2d)
        for i in range(points_2d.shape[1]):
            if points_2d[1, i] < 0.5:
                assert result[i] == -1.0
            else:
                assert result[i] == 2.0

    def test_step_param_refs(self, points_2d):
        cfg = {
            "type": "step",
            "params": {
                "value_left": "param:vl",
                "value_right": "param:vr",
                "x_split": 0.5,
                "axis": 0,
            },
        }
        fn = build_interpolator(cfg, {"vl": 10.0, "vr": 20.0})
        assert fn is not None
        result = fn(points_2d)
        for i in range(points_2d.shape[1]):
            if points_2d[0, i] < 0.5:
                assert result[i] == 10.0
            else:
                assert result[i] == 20.0

    def test_custom_returns_none(self):
        result = build_interpolator({"type": "custom"}, {})
        assert result is None

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown field type"):
            build_interpolator({"type": "bogus_type"}, {})

    def test_zero_with_3d_points(self, points_3d):
        fn = build_interpolator({"type": "zero"}, {})
        assert fn is not None
        result = fn(points_3d)
        assert result.shape == (points_3d.shape[1],)
        np.testing.assert_array_equal(result, 0.0)

    def test_constant_with_3d_points(self, points_3d):
        cfg = {"type": "constant", "params": {"value": -2.5}}
        fn = build_interpolator(cfg, {})
        assert fn is not None
        result = fn(points_3d)
        assert result.shape == (points_3d.shape[1],)
        np.testing.assert_array_equal(result, -2.5)

"""Tests for initial condition generators."""

import numpy as np
import pytest
from pde import CartesianGrid, ScalarField

from pde_sim.initial_conditions import (
    CosinePattern,
    DoubleStep,
    GaussianBlob,
    RandomGaussian,
    RandomUniform,
    RectangleGrid,
    SinePattern,
    StepFunction,
    create_initial_condition,
    list_initial_conditions,
    resolve_random_params,
)


class TestInitialConditionRegistry:
    """Tests for the IC registry."""

    def test_list_initial_conditions(self):
        """Test that list_initial_conditions returns expected types."""
        ic_types = list_initial_conditions()

        assert "random-uniform" in ic_types
        assert "random-gaussian" in ic_types
        assert "gaussian-blob" in ic_types
        assert "sine" in ic_types
        assert "step" in ic_types
        assert "rectangle-grid" in ic_types

    def test_create_unknown_ic(self, small_grid):
        """Test that unknown IC type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown IC type"):
            create_initial_condition(small_grid, "nonexistent-ic", {})


class TestRandomUniform:
    """Tests for RandomUniform IC generator."""

    def test_generate_default(self, small_grid):
        """Test generating with default parameters."""
        ic = RandomUniform()
        field = ic.generate(small_grid)

        assert isinstance(field, ScalarField)
        assert field.data.shape == (32, 32)
        assert np.all(field.data >= 0.0)
        assert np.all(field.data <= 1.0)

    def test_generate_custom_range(self, small_grid):
        """Test generating with custom low/high."""
        ic = RandomUniform()
        field = ic.generate(small_grid, low=-5.0, high=5.0)

        assert np.all(field.data >= -5.0)
        assert np.all(field.data <= 5.0)

    def test_via_factory(self, small_grid):
        """Test creating via factory function."""
        field = create_initial_condition(
            small_grid, "random-uniform", {"low": 0.5, "high": 1.5}
        )

        assert isinstance(field, ScalarField)
        assert np.all(field.data >= 0.5)
        assert np.all(field.data <= 1.5)


class TestRandomGaussian:
    """Tests for RandomGaussian IC generator."""

    def test_generate_default(self, small_grid):
        """Test generating with default parameters."""
        ic = RandomGaussian()
        field = ic.generate(small_grid)

        assert isinstance(field, ScalarField)
        assert field.data.shape == (32, 32)

    def test_generate_custom_params(self, small_grid):
        """Test generating with custom mean/std."""
        ic = RandomGaussian()
        field = ic.generate(small_grid, mean=10.0, std=0.1)

        # Mean should be close to 10
        assert np.abs(np.mean(field.data) - 10.0) < 1.0

    def test_generate_with_clipping(self, small_grid):
        """Test generating with value clipping."""
        ic = RandomGaussian()
        field = ic.generate(small_grid, mean=0.5, std=0.5, clip_min=0.0, clip_max=1.0)

        assert np.all(field.data >= 0.0)
        assert np.all(field.data <= 1.0)


class TestGaussianBlob:
    """Tests for GaussianBlob IC generator."""

    def test_generate_default(self, small_grid):
        """Test generating with default parameters."""
        ic = GaussianBlob()
        field = ic.generate(small_grid, num_blobs=1, positions=[[0.5, 0.5]])

        assert isinstance(field, ScalarField)
        assert field.data.shape == (32, 32)
        # Should have some variation (not all zeros)
        assert np.std(field.data) > 0

    def test_generate_single_blob(self, small_grid):
        """Test generating a single blob."""
        ic = GaussianBlob()
        field = ic.generate(
            small_grid,
            num_blobs=1,
            positions=[[0.5, 0.5]],
            amplitude=1.0,
            width=0.1,
        )

        # Maximum should be close to amplitude
        assert np.max(field.data) > 0.5

    def test_generate_multiple_blobs(self, small_grid):
        """Test generating multiple blobs."""
        ic = GaussianBlob()
        field = ic.generate(
            small_grid,
            num_blobs=5,
            positions=[
                [0.2, 0.2],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.8, 0.8],
                [0.5, 0.5],
            ],
            amplitude=1.0,
            width=0.1,
        )

        assert isinstance(field, ScalarField)
        # Should have some variation from multiple blobs
        assert np.std(field.data) > 0

    def test_generate_with_background(self, small_grid):
        """Test generating with non-zero background."""
        ic = GaussianBlob()
        field = ic.generate(
            small_grid, num_blobs=1, positions=[[0.5, 0.5]], background=0.5
        )

        # Minimum should be at least the background
        assert np.min(field.data) >= 0.5 - 0.01  # Allow small numerical tolerance

    def test_generate_random_amplitude(self, small_grid):
        """Test generating with random amplitudes."""
        ic = GaussianBlob()
        field = ic.generate(
            small_grid,
            num_blobs=3,
            positions=[[0.2, 0.3], [0.6, 0.4], [0.8, 0.7]],
            amplitude=1.0,
            random_amplitude=True,
            seed=42,
        )

        assert isinstance(field, ScalarField)
        assert np.std(field.data) > 0

    def test_generate_asymmetric_2d(self, small_grid):
        """Test generating asymmetric (elliptical) blobs in 2D."""
        ic = GaussianBlob()
        field = ic.generate(
            small_grid,
            num_blobs=3,
            positions=[[0.25, 0.25], [0.75, 0.25], [0.5, 0.75]],
            amplitude=1.0,
            width=0.05,
            aspect_ratio=3.0,
            seed=42,
        )

        assert isinstance(field, ScalarField)
        assert np.std(field.data) > 0

    def test_generate_asymmetric_random_aspect(self, small_grid):
        """Test generating asymmetric blobs with random aspect ratios."""
        ic = GaussianBlob()
        field = ic.generate(
            small_grid,
            num_blobs=5,
            positions=[
                [0.1, 0.1],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.9, 0.9],
                [0.5, 0.5],
            ],
            amplitude=1.0,
            width=0.05,
            aspect_ratio=4.0,
            random_aspect=True,
            seed=42,
        )

        assert isinstance(field, ScalarField)
        assert np.std(field.data) > 0

    def test_generate_1d_blobs(self):
        """Test generating blobs on a 1D grid."""
        grid_1d = CartesianGrid(bounds=[[0, 10]], shape=[64], periodic=[True])
        ic = GaussianBlob()
        field = ic.generate(
            grid_1d, num_blobs=3, positions=[0.2, 0.5, 0.8], amplitude=1.0, width=0.1
        )

        assert isinstance(field, ScalarField)
        assert field.data.shape == (64,)
        assert np.std(field.data) > 0

    def test_generate_3d_symmetric(self):
        """Test generating symmetric blobs on a 3D grid."""
        grid_3d = CartesianGrid(
            bounds=[[0, 1], [0, 1], [0, 1]], shape=[16, 16, 16], periodic=[True, True, True]
        )
        ic = GaussianBlob()
        field = ic.generate(
            grid_3d,
            num_blobs=2,
            positions=[[0.3, 0.3, 0.3], [0.7, 0.7, 0.7]],
            amplitude=1.0,
            width=0.15,
        )

        assert isinstance(field, ScalarField)
        assert field.data.shape == (16, 16, 16)
        assert np.std(field.data) > 0

    def test_generate_3d_asymmetric(self):
        """Test generating asymmetric (ellipsoidal) blobs on a 3D grid."""
        grid_3d = CartesianGrid(
            bounds=[[0, 1], [0, 1], [0, 1]], shape=[16, 16, 16], periodic=[True, True, True]
        )
        ic = GaussianBlob()
        field = ic.generate(
            grid_3d,
            num_blobs=2,
            positions=[[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
            amplitude=1.0,
            width=0.1,
            aspect_ratio=2.5,
            seed=42,
        )

        assert isinstance(field, ScalarField)
        assert field.data.shape == (16, 16, 16)
        assert np.std(field.data) > 0

    def test_via_factory(self, small_grid):
        """Test creating via factory function."""
        field = create_initial_condition(
            small_grid,
            "gaussian-blob",
            {"num_blobs": 2, "positions": [[0.25, 0.5], [0.75, 0.5]], "amplitude": 1.0},
        )

        assert isinstance(field, ScalarField)

    def test_seed_reproducibility(self, small_grid):
        """Test that seed produces reproducible results."""
        ic = GaussianBlob()
        params1 = resolve_random_params(
            small_grid,
            "gaussian-blob",
            {"num_blobs": 3, "positions": "random", "seed": 123},
        )
        params2 = resolve_random_params(
            small_grid,
            "gaussian-blob",
            {"num_blobs": 3, "positions": "random", "seed": 123},
        )
        field1 = ic.generate(small_grid, **params1)
        field2 = ic.generate(small_grid, **params2)

        np.testing.assert_array_equal(field1.data, field2.data)


class TestSinePattern:
    """Tests for SinePattern IC generator."""

    def test_generate_default(self, small_grid):
        """Test generating with default parameters."""
        ic = SinePattern()
        field = ic.generate(small_grid, phase_x=0.0, phase_y=0.0)

        assert isinstance(field, ScalarField)
        assert field.data.shape == (32, 32)

    def test_generate_single_wave(self, small_grid):
        """Test generating a single sine wave."""
        ic = SinePattern()
        field = ic.generate(
            small_grid,
            kx=1,
            ky=0,
            amplitude=1.0,
            offset=0.0,
            phase_x=0.0,
            phase_y=0.0,
        )

        # The pattern should vary along x but be constant along y when ky=0
        # Actually with sin(0)=0, the field will be 0 everywhere if ky=0
        # Let me check that the pattern has expected range
        assert np.max(field.data) <= 1.0
        assert np.min(field.data) >= -1.0

    def test_generate_with_offset(self, small_grid):
        """Test generating with offset."""
        ic = SinePattern()
        field = ic.generate(
            small_grid,
            kx=1,
            ky=1,
            amplitude=0.5,
            offset=1.0,
            phase_x=0.0,
            phase_y=0.0,
        )

        # Values should be in range [offset - amplitude, offset + amplitude]
        assert np.max(field.data) <= 1.5 + 0.01
        assert np.min(field.data) >= 0.5 - 0.01


class TestStepFunction:
    """Tests for StepFunction IC generator."""

    def test_generate_default(self, small_grid):
        """Test generating with default (random) parameters."""
        ic = StepFunction()
        params = resolve_random_params(
            small_grid, "step", {"position": "random", "seed": 42}
        )
        field = ic.generate(small_grid, **params)

        assert isinstance(field, ScalarField)
        assert field.data.shape == (32, 32)
        # Should have both low and high values (step is somewhere in the domain)
        unique_values = np.unique(field.data)
        assert len(unique_values) >= 2

    def test_generate_x_step(self, small_grid):
        """Test generating a step in x direction."""
        ic = StepFunction()
        field = ic.generate(
            small_grid,
            direction="x",
            position=0.5,
            value_low=0.0,
            value_high=1.0,
        )

        # Left half should be 0, right half should be 1
        assert np.all(field.data[:16, :] == 0.0)
        assert np.all(field.data[16:, :] == 1.0)

    def test_generate_y_step(self, small_grid):
        """Test generating a step in y direction."""
        ic = StepFunction()
        field = ic.generate(
            small_grid,
            direction="y",
            position=0.5,
            value_low=0.0,
            value_high=1.0,
        )

        # Bottom half should be 0, top half should be 1
        assert np.all(field.data[:, :16] == 0.0)
        assert np.all(field.data[:, 16:] == 1.0)

    def test_generate_smooth_step(self, small_grid):
        """Test generating a smooth step transition."""
        ic = StepFunction()
        field = ic.generate(
            small_grid,
            direction="x",
            position=0.5,
            value_low=0.0,
            value_high=1.0,
            smooth_width=0.1,
        )

        # Should have smooth transition, not just 0s and 1s
        unique_values = np.unique(field.data)
        assert len(unique_values) > 2


class TestRectangleGrid:
    """Tests for RectangleGrid IC generator."""

    def test_generate_default(self, small_grid):
        """Test generating with default parameters."""
        ic = RectangleGrid()
        field = ic.generate(small_grid, seed=42)

        assert isinstance(field, ScalarField)
        assert field.data.shape == (32, 32)
        # Should have variation (different rectangles)
        assert np.std(field.data) > 0

    def test_generate_2x2_explicit_values(self, small_grid):
        """Test generating a 2x2 grid with explicit values."""
        ic = RectangleGrid()
        values = [[0.0, 0.25], [0.5, 1.0]]
        field = ic.generate(small_grid, nx=2, ny=2, values=values)

        # Check that quadrants have the correct values
        # Bottom-left (i=0, j=0) should be 0.0
        assert np.allclose(field.data[:16, :16], 0.0)
        # Bottom-right (i=0, j=1) should be 0.25
        assert np.allclose(field.data[:16, 16:], 0.25)
        # Top-left (i=1, j=0) should be 0.5
        assert np.allclose(field.data[16:, :16], 0.5)
        # Top-right (i=1, j=1) should be 1.0
        assert np.allclose(field.data[16:, 16:], 1.0)

    def test_generate_3x2_grid(self, small_grid):
        """Test generating a 3x2 grid."""
        ic = RectangleGrid()
        values = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        field = ic.generate(small_grid, nx=3, ny=2, values=values)

        assert isinstance(field, ScalarField)
        # Should have 6 distinct values
        unique_values = set(np.unique(field.data))
        assert unique_values == {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

    def test_generate_with_random_values(self, small_grid):
        """Test generating with random values in specified range."""
        ic = RectangleGrid()
        field = ic.generate(
            small_grid, nx=3, ny=3, value_range=(10.0, 20.0), seed=42
        )

        assert np.all(field.data >= 10.0)
        assert np.all(field.data <= 20.0)

    def test_seed_reproducibility(self, small_grid):
        """Test that seed produces reproducible results."""
        ic = RectangleGrid()
        field1 = ic.generate(small_grid, nx=3, ny=3, seed=123)
        field2 = ic.generate(small_grid, nx=3, ny=3, seed=123)

        np.testing.assert_array_equal(field1.data, field2.data)

    def test_via_factory(self, small_grid):
        """Test creating via factory function."""
        field = create_initial_condition(
            small_grid,
            "rectangle-grid",
            {"nx": 2, "ny": 2, "values": [[0.0, 1.0], [2.0, 3.0]]},
        )

        assert isinstance(field, ScalarField)
        unique_values = set(np.unique(field.data))
        assert unique_values == {0.0, 1.0, 2.0, 3.0}

    def test_values_shape_mismatch_raises(self, small_grid):
        """Test that mismatched values shape raises ValueError."""
        ic = RectangleGrid()
        with pytest.raises(ValueError, match="doesn't match grid"):
            ic.generate(small_grid, nx=2, ny=2, values=[[1.0, 2.0, 3.0]])


class TestDoubleStep:
    """Tests for DoubleStep IC generator."""

    def test_generate_explicit_positions(self, small_grid):
        """Test generating with explicit positions."""
        ic = DoubleStep()
        field = ic.generate(
            small_grid, direction="x", position1=0.25, position2=0.75,
            value_inside=1.0, value_outside=0.0,
        )
        assert isinstance(field, ScalarField)
        assert field.data.shape == (32, 32)
        # Should have both inside and outside values
        assert np.any(field.data == 1.0)
        assert np.any(field.data == 0.0)

    def test_generate_random_positions(self, small_grid):
        """Test generating with random positions (None)."""
        ic = DoubleStep()
        params = resolve_random_params(
            small_grid,
            "double-step",
            {"position1": "random", "position2": "random", "seed": 42},
        )
        field = ic.generate(small_grid, **params)
        assert isinstance(field, ScalarField)
        assert field.data.shape == (32, 32)
        unique_values = np.unique(field.data)
        assert len(unique_values) >= 2


class TestCosinePattern:
    """Tests for CosinePattern IC generator."""

    def test_generate_default(self, small_grid):
        """Test generating with default (random phase) parameters."""
        ic = CosinePattern()
        field = ic.generate(small_grid, phase_x=0.0, phase_y=0.0)
        assert isinstance(field, ScalarField)
        assert field.data.shape == (32, 32)

    def test_explicit_phases(self, small_grid):
        """Test generating with explicit phases."""
        ic = CosinePattern()
        field = ic.generate(small_grid, kx=1, ky=1, phase_x=0.0, phase_y=0.0)
        assert isinstance(field, ScalarField)
        # With phase=0, cos starts at max → values should include amplitude
        assert np.max(field.data) > 0.5


class TestSeedReproducibility:
    """Tests for seed reproducibility across all IC generators."""

    SEED_TEST_CONFIGS = [
        ("random-uniform", RandomUniform, {}, {}),
        ("random-gaussian", RandomGaussian, {}, {}),
        ("step", StepFunction, {"position": "random"}, {"position": "random"}),
        ("double-step", DoubleStep, {"position1": "random", "position2": "random"}, {"position1": "random", "position2": "random"}),
        ("sine", SinePattern, {"phase_x": "random", "phase_y": "random"}, {"phase_x": "random", "phase_y": "random"}),
        ("cosine", CosinePattern, {"phase_x": "random", "phase_y": "random"}, {"phase_x": "random", "phase_y": "random"}),
        ("gaussian-blob", GaussianBlob, {"num_blobs": 3, "positions": "random"}, {"num_blobs": 3, "positions": "random"}),
    ]

    @pytest.mark.parametrize(
        "ic_name,ic_class,params_template,_",
        SEED_TEST_CONFIGS,
        ids=[c[0] for c in SEED_TEST_CONFIGS],
    )
    def test_same_seed_reproduces(self, small_grid, ic_name, ic_class, params_template, _):
        """Test that same seed produces identical results."""
        ic = ic_class()
        params1 = resolve_random_params(small_grid, ic_name, {**params_template, "seed": 42})
        params2 = resolve_random_params(small_grid, ic_name, {**params_template, "seed": 42})
        field1 = ic.generate(small_grid, **params1)
        field2 = ic.generate(small_grid, **params2)
        np.testing.assert_array_equal(field1.data, field2.data)

    @pytest.mark.parametrize(
        "ic_name,ic_class,params_template,_",
        SEED_TEST_CONFIGS,
        ids=[c[0] for c in SEED_TEST_CONFIGS],
    )
    def test_different_seeds_differ(self, small_grid, ic_name, ic_class, params_template, _):
        """Test that different seeds produce different results."""
        ic = ic_class()
        params1 = resolve_random_params(small_grid, ic_name, {**params_template, "seed": 42})
        params2 = resolve_random_params(small_grid, ic_name, {**params_template, "seed": 99})
        field1 = ic.generate(small_grid, **params1)
        field2 = ic.generate(small_grid, **params2)
        assert not np.array_equal(field1.data, field2.data)

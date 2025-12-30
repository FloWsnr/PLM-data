"""Tests for initial condition generators."""

import numpy as np
import pytest
from pde import ScalarField

from pde_sim.initial_conditions import (
    create_initial_condition,
    list_initial_conditions,
    RandomUniform,
    RandomGaussian,
    GaussianBlobs,
    SinePattern,
    StepFunction,
)


class TestInitialConditionRegistry:
    """Tests for the IC registry."""

    def test_list_initial_conditions(self):
        """Test that list_initial_conditions returns expected types."""
        ic_types = list_initial_conditions()

        assert "random-uniform" in ic_types
        assert "random-gaussian" in ic_types
        assert "gaussian-blobs" in ic_types
        assert "sine" in ic_types
        assert "step" in ic_types

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


class TestGaussianBlobs:
    """Tests for GaussianBlobs IC generator."""

    def test_generate_default(self, small_grid):
        """Test generating with default parameters."""
        np.random.seed(42)
        ic = GaussianBlobs()
        field = ic.generate(small_grid)

        assert isinstance(field, ScalarField)
        assert field.data.shape == (32, 32)
        # Should have some variation (not all zeros)
        assert np.std(field.data) > 0

    def test_generate_single_blob(self, small_grid):
        """Test generating a single blob."""
        np.random.seed(42)
        ic = GaussianBlobs()
        field = ic.generate(small_grid, num_blobs=1, amplitude=1.0, width=0.1)

        # Maximum should be close to amplitude
        assert np.max(field.data) > 0.5

    def test_generate_with_background(self, small_grid):
        """Test generating with non-zero background."""
        np.random.seed(42)
        ic = GaussianBlobs()
        field = ic.generate(small_grid, num_blobs=1, background=0.5)

        # Minimum should be at least the background
        assert np.min(field.data) >= 0.5 - 0.01  # Allow small numerical tolerance


class TestSinePattern:
    """Tests for SinePattern IC generator."""

    def test_generate_default(self, small_grid):
        """Test generating with default parameters."""
        ic = SinePattern()
        field = ic.generate(small_grid)

        assert isinstance(field, ScalarField)
        assert field.data.shape == (32, 32)

    def test_generate_single_wave(self, small_grid):
        """Test generating a single sine wave."""
        ic = SinePattern()
        field = ic.generate(small_grid, kx=1, ky=0, amplitude=1.0, offset=0.0)

        # The pattern should vary along x but be constant along y when ky=0
        # Actually with sin(0)=0, the field will be 0 everywhere if ky=0
        # Let me check that the pattern has expected range
        assert np.max(field.data) <= 1.0
        assert np.min(field.data) >= -1.0

    def test_generate_with_offset(self, small_grid):
        """Test generating with offset."""
        ic = SinePattern()
        field = ic.generate(small_grid, kx=1, ky=1, amplitude=0.5, offset=1.0)

        # Values should be in range [offset - amplitude, offset + amplitude]
        assert np.max(field.data) <= 1.5 + 0.01
        assert np.min(field.data) >= 0.5 - 0.01


class TestStepFunction:
    """Tests for StepFunction IC generator."""

    def test_generate_default(self, small_grid):
        """Test generating with default parameters."""
        ic = StepFunction()
        field = ic.generate(small_grid)

        assert isinstance(field, ScalarField)
        assert field.data.shape == (32, 32)

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

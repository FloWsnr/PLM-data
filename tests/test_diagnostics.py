"""Tests for trajectory stagnation detection."""

import numpy as np
import pytest

from pde_sim.core.diagnostics import check_trajectory_stagnation


def _identity_extract(state, field_name):
    """Simple extractor that treats state as a dict of arrays."""
    return state[field_name]


class TestCheckTrajectoryStagnation:
    """Tests for the check_trajectory_stagnation function."""

    def test_completely_constant(self):
        """All frames identical -> stagnant from frame 0."""
        data = np.ones((10, 10))
        frames = [{"u": data.copy()} for _ in range(20)]

        result = check_trajectory_stagnation(
            all_fields=frames,
            field_names=["u"],
            extract_field_fn=_identity_extract,
        )

        assert result["stagnant_fields"] == ["u"]
        info = result["fields"]["u"]
        assert info["stagnant"] is True
        assert info["stagnant_from_frame"] == 0
        assert info["trailing_stagnant_frames"] == 20
        assert info["field_range"] == 0.0
        assert info["max_relative_change"] == 0.0
        assert info["final_relative_change"] == 0.0

    def test_quick_convergence(self):
        """First few frames active, then flat -> warns with correct stagnant_from_frame."""
        frames = []
        num_active = 5
        num_stagnant = 15
        num_total = num_active + num_stagnant

        # Active phase: increasing values
        for i in range(num_active):
            frames.append({"u": np.full((10, 10), float(i))})

        # Stagnant phase: all frames have the same value
        steady = np.full((10, 10), float(num_active - 1))
        for _ in range(num_stagnant):
            frames.append({"u": steady.copy()})

        result = check_trajectory_stagnation(
            all_fields=frames,
            field_names=["u"],
            extract_field_fn=_identity_extract,
        )

        assert result["stagnant_fields"] == ["u"]
        info = result["fields"]["u"]
        assert info["stagnant"] is True
        # Stagnant streak = 15 frame pairs (from frame 4 onwards, changes are 0)
        # Actually, relative_changes has 19 entries (num_total - 1)
        # Changes 0..3 are active (value 1.0/4.0 = 0.25 each)
        # Changes 4..18 are zero -> trailing_stagnant = 15
        assert info["trailing_stagnant_frames"] == 15
        # stagnant_from_frame = num_total - 1 - trailing_stagnant = 19 - 15 = 4
        assert info["stagnant_from_frame"] == 4
        assert info["field_range"] == 4.0  # max=4, min=0

    def test_active_trajectory(self):
        """All frames meaningfully different -> no warning."""
        np.random.seed(42)
        frames = []
        for i in range(20):
            # Each frame has substantially different values
            frames.append({"u": np.random.rand(10, 10) * (i + 1)})

        result = check_trajectory_stagnation(
            all_fields=frames,
            field_names=["u"],
            extract_field_fn=_identity_extract,
        )

        assert result["stagnant_fields"] == []
        assert result["fields"]["u"]["stagnant"] is False
        assert result["fields"]["u"]["stagnant_from_frame"] is None

    def test_multi_field_partial_stagnation(self):
        """One field stagnates, another stays active -> only warns for stagnant field."""
        frames = []
        np.random.seed(123)
        for i in range(20):
            frames.append({
                "u": np.full((10, 10), 1.0),  # Constant -> stagnant
                "v": np.random.rand(10, 10) * (i + 1),  # Active
            })

        result = check_trajectory_stagnation(
            all_fields=frames,
            field_names=["u", "v"],
            extract_field_fn=_identity_extract,
        )

        assert result["stagnant_fields"] == ["u"]
        assert result["fields"]["u"]["stagnant"] is True
        assert result["fields"]["v"]["stagnant"] is False

    def test_zero_range_field(self):
        """Field uniformly zero across space and time -> stagnant with zero range."""
        frames = [{"u": np.zeros((10, 10))} for _ in range(10)]

        result = check_trajectory_stagnation(
            all_fields=frames,
            field_names=["u"],
            extract_field_fn=_identity_extract,
        )

        assert result["stagnant_fields"] == ["u"]
        info = result["fields"]["u"]
        assert info["stagnant"] is True
        assert info["field_range"] == 0.0
        assert info["stagnant_from_frame"] == 0

    def test_threshold_sensitivity(self):
        """Changes just below threshold are detected as stagnant."""
        field_range = 100.0  # Will have range of 100
        frames = []
        # Frame 0: establishes the range
        data = np.zeros((10, 10))
        data[0, 0] = field_range
        frames.append({"u": data.copy()})

        # Remaining frames: tiny changes well below default threshold (1e-4 * 100 = 0.01)
        for i in range(1, 20):
            data_i = data.copy()
            data_i[5, 5] += 1e-4  # max change = 1e-4, relative = 1e-6 << 1e-4
            frames.append({"u": data_i})

        result = check_trajectory_stagnation(
            all_fields=frames,
            field_names=["u"],
            extract_field_fn=_identity_extract,
        )

        assert result["stagnant_fields"] == ["u"]
        assert result["fields"]["u"]["stagnant"] is True

    def test_custom_threshold(self):
        """Custom rel_threshold changes what counts as stagnant."""
        frames = []
        for i in range(20):
            frames.append({"u": np.full((10, 10), float(i) * 0.01)})

        # With default threshold (1e-4), changes of 0.01 in range 0.19 => ~0.053 >> 1e-4
        result_default = check_trajectory_stagnation(
            all_fields=frames,
            field_names=["u"],
            extract_field_fn=_identity_extract,
        )
        assert result_default["stagnant_fields"] == []

        # With very high threshold, everything looks stagnant
        result_high = check_trajectory_stagnation(
            all_fields=frames,
            field_names=["u"],
            extract_field_fn=_identity_extract,
            rel_threshold=0.1,
        )
        assert result_high["stagnant_fields"] == ["u"]

    def test_min_stagnant_fraction(self):
        """Stagnant streak too short for the fraction -> no warning."""
        frames = []
        # 5 active frames, then 3 stagnant frames (total 8)
        for i in range(5):
            frames.append({"u": np.full((10, 10), float(i))})
        steady = np.full((10, 10), 4.0)
        for _ in range(3):
            frames.append({"u": steady.copy()})

        # min_stagnant_fraction=0.5 means we need 4 stagnant frames, but we only have 3
        result = check_trajectory_stagnation(
            all_fields=frames,
            field_names=["u"],
            extract_field_fn=_identity_extract,
            min_stagnant_fraction=0.5,
        )

        assert result["stagnant_fields"] == []
        assert result["fields"]["u"]["stagnant"] is False
        assert result["fields"]["u"]["trailing_stagnant_frames"] == 3

    def test_1d_data(self):
        """Works with 1D spatial data."""
        frames = [{"u": np.ones(50)} for _ in range(10)]

        result = check_trajectory_stagnation(
            all_fields=frames,
            field_names=["u"],
            extract_field_fn=_identity_extract,
        )

        assert result["stagnant_fields"] == ["u"]
        assert result["fields"]["u"]["stagnant"] is True

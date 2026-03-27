"""Tests for post-run stagnation diagnostics."""

import numpy as np
import pytest

from plm_data.core.diagnostics import build_stagnation_report


def test_constant_field_is_reported_as_stagnant():
    frames = {"u": [np.ones((8, 8)) for _ in range(20)]}

    report = build_stagnation_report(frames)

    assert report["applied"] is True
    assert report["reason"] is None
    assert report["checked_fields"] == ["u"]
    assert report["skipped_static_fields"] == []
    assert report["stagnant_fields"] == ["u"]
    assert report["fields"]["u"] == {
        "stagnant": True,
        "stagnant_from_frame": 0,
        "trailing_stagnant_frames": 20,
        "field_range": 0.0,
        "max_relative_change": 0.0,
        "final_relative_change": 0.0,
    }


def test_converged_trailing_tail_is_reported_as_stagnant():
    frames = {
        "u": [np.full((8, 8), float(i)) for i in range(5)]
        + [np.full((8, 8), 4.0) for _ in range(15)]
    }

    report = build_stagnation_report(frames)

    assert report["stagnant_fields"] == ["u"]
    info = report["fields"]["u"]
    assert info["stagnant"] is True
    assert info["stagnant_from_frame"] == 4
    assert info["trailing_stagnant_frames"] == 15
    assert info["field_range"] == 4.0
    assert info["max_relative_change"] == pytest.approx(0.25)
    assert info["final_relative_change"] == 0.0


def test_active_trajectory_is_not_reported_as_stagnant():
    rng = np.random.default_rng(42)
    frames = {"u": [rng.random((8, 8)) * (i + 1) for i in range(20)]}

    report = build_stagnation_report(frames)

    assert report["stagnant_fields"] == []
    assert report["fields"]["u"]["stagnant"] is False
    assert report["fields"]["u"]["stagnant_from_frame"] is None


def test_insufficient_frames_disables_stagnation_analysis():
    frames = {"u": [np.ones((8, 8))]}

    report = build_stagnation_report(frames, skipped_static_fields=["pressure"])

    assert report["applied"] is False
    assert report["reason"] == "insufficient_frames"
    assert report["checked_fields"] == ["u"]
    assert report["skipped_static_fields"] == ["pressure"]
    assert report["stagnant_fields"] == []
    assert report["fields"] == {}


def test_complex_phase_change_is_not_treated_as_constant():
    frames = {
        "electric_field_x": [
            np.full((4, 4), np.exp(1j * theta), dtype=np.complex128)
            for theta in (0.0, np.pi / 4.0, np.pi / 2.0)
        ]
    }

    report = build_stagnation_report(frames)

    assert report["applied"] is True
    assert report["stagnant_fields"] == []
    assert report["fields"]["electric_field_x"]["stagnant"] is False

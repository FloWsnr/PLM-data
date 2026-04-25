"""Tests for plm_data.sampling.values."""

from plm_data.sampling.values import contains_sampler_spec


def test_contains_sampler_spec_searches_nested_containers():
    assert contains_sampler_spec({"alpha": [{"sample": "uniform"}]}) is True
    assert contains_sampler_spec({"alpha": [1.0, {"beta": 2.0}]}) is False

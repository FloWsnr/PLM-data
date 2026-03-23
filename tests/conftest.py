"""Shared test fixtures."""

from pathlib import Path

import pytest

from plm_data.core.config import (
    DomainConfig,
    ICConfig,
    OutputConfig,
    SimulationConfig,
    SolverConfig,
)


@pytest.fixture
def rectangle_domain():
    return DomainConfig(
        type="rectangle",
        params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
    )


@pytest.fixture
def direct_solver():
    return SolverConfig(options={"ksp_type": "preonly", "pc_type": "lu"})


@pytest.fixture
def heat_config(tmp_path, rectangle_domain, direct_solver):
    return SimulationConfig(
        preset="heat",
        parameters={"kappa": 0.01},
        domain=rectangle_domain,
        output_resolution=[4, 4],
        initial_condition=ICConfig(
            type="gaussian_bump",
            params={"sigma": 0.1, "amplitude": 1.0, "cx": 0.5, "cy": 0.5},
        ),
        output=OutputConfig(
            path=tmp_path,
            num_frames=2,
            formats=["numpy"],
        ),
        solver=direct_solver,
        dt=0.01,
        t_end=0.01,
        seed=42,
    )


@pytest.fixture
def poisson_config(tmp_path, rectangle_domain, direct_solver):
    return SimulationConfig(
        preset="poisson",
        parameters={"kappa": 1.0, "f_amplitude": 1.0},
        domain=rectangle_domain,
        output_resolution=[4, 4],
        initial_condition=None,
        output=OutputConfig(
            path=tmp_path,
            num_frames=1,
            formats=["numpy"],
        ),
        solver=direct_solver,
        seed=42,
    )

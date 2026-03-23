"""Shared test fixtures."""

from pathlib import Path

import pytest

from plm_data.core.config import (
    BCConfig,
    DomainConfig,
    ICConfig,
    OutputConfig,
    SimulationConfig,
    SolverConfig,
    SourceTermConfig,
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
        boundary_conditions={
            "u": {
                "x-": BCConfig(type="neumann", value=0.0),
                "x+": BCConfig(type="neumann", value=0.0),
                "y-": BCConfig(type="neumann", value=0.0),
                "y+": BCConfig(type="neumann", value=0.0),
            },
        },
        source_terms={"u": SourceTermConfig(type="none", params={})},
        initial_conditions={
            "u": ICConfig(
                type="gaussian_bump",
                params={"sigma": 0.1, "amplitude": 1.0, "cx": 0.5, "cy": 0.5},
            ),
        },
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
def poisson_config(tmp_path, direct_solver):
    domain = DomainConfig(
        type="rectangle",
        params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
    )
    return SimulationConfig(
        preset="poisson",
        parameters={"kappa": 1.0, "f_amplitude": 1.0},
        domain=domain,
        output_resolution=[4, 4],
        boundary_conditions={
            "u": {
                "x-": BCConfig(type="dirichlet", value=0.0),
                "x+": BCConfig(type="dirichlet", value=0.0),
                "y-": BCConfig(type="dirichlet", value=0.0),
                "y+": BCConfig(type="dirichlet", value=0.0),
            },
        },
        source_terms={
            "u": SourceTermConfig(
                type="sine_product",
                params={"amplitude": "param:f_amplitude", "kx": 1, "ky": 1},
            ),
        },
        initial_conditions={},
        output=OutputConfig(
            path=tmp_path,
            num_frames=1,
            formats=["numpy"],
        ),
        solver=direct_solver,
        seed=42,
    )

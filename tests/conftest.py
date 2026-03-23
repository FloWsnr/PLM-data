"""Shared test fixtures."""

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
def cahn_hilliard_config(tmp_path, rectangle_domain, direct_solver):
    return SimulationConfig(
        preset="cahn_hilliard",
        parameters={"lmbda": 0.01, "barrier_height": 100.0, "mobility": 1.0, "theta": 0.5},
        domain=rectangle_domain,
        output_resolution=[4, 4],
        boundary_conditions={"c": {}},
        source_terms={"c": SourceTermConfig(type="none", params={})},
        initial_conditions={
            "c": ICConfig(
                type="random_perturbation",
                params={"mean": 0.63, "std": 0.02},
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            num_frames=2,
            formats=["numpy"],
        ),
        solver=SolverConfig(
            options={
                "snes_type": "newtonls",
                "snes_linesearch_type": "none",
                "ksp_type": "preonly",
                "pc_type": "lu",
            }
        ),
        dt=5e-6,
        t_end=5e-6,
        seed=42,
    )


@pytest.fixture
def navier_stokes_config(tmp_path, direct_solver):
    domain = DomainConfig(
        type="rectangle",
        params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
    )
    return SimulationConfig(
        preset="navier_stokes",
        parameters={"Re": 25.0, "k": 1.0},
        domain=domain,
        output_resolution=[4, 4],
        boundary_conditions={
            "velocity": {
                "x-": BCConfig(type="dirichlet", value=[0.0, 0.0]),
                "x+": BCConfig(type="dirichlet", value=[0.0, 0.0]),
                "y-": BCConfig(type="dirichlet", value=[0.0, 0.0]),
                "y+": BCConfig(type="dirichlet", value=[1.0, 0.0]),
            },
            "pressure": {},
        },
        source_terms={
            "velocity": SourceTermConfig(type="none", params={}),
            "pressure": SourceTermConfig(type="none", params={}),
        },
        initial_conditions={
            "velocity": ICConfig(type="custom", params={}),
        },
        output=OutputConfig(
            path=tmp_path,
            num_frames=2,
            formats=["numpy"],
        ),
        solver=SolverConfig(
            options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_14": "80",
                "mat_mumps_icntl_24": "1",
                "mat_mumps_icntl_25": "0",
                "ksp_error_if_not_converged": "1",
            }
        ),
        dt=0.1,
        t_end=0.1,
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

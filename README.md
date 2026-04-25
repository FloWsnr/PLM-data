# PLM-data

Generate random two-dimensional, time-dependent PDE simulation datasets using
[DOLFINx](https://github.com/FEniCS/dolfinx) and Gmsh.

## Installation

Create the default conda environment with DOLFINx and the project tools:

```bash
conda create -n fenicsx-env -c conda-forge fenics-dolfinx mpich python=3.12 ffmpeg python-gmsh
conda activate fenicsx-env
conda install -c conda-forge dolfinx_mpc
pip install pytest pytest-xdist pyright matplotlib
```

## Usage

`run.sh` is the user-facing entrypoint. Every run requires an explicit seed.

```bash
./run.sh --seed 1234
./run.sh -n 4 --seed 1234
./run.sh -n 4 --core-list 0-3 --seed 1234
```

The wrapper activates `fenicsx-env`, starts one single-threaded MPI rank per
requested physical core, and writes outputs under `./output` by default. Use
`--output-dir` to choose a different root.

Output directories include the sampled PDE, sampled domain, seed, and short run
id:

```text
output/<pde>/<domain>/seed_<seed>_<run-id>/
```

Default random-run formats are:

- `numpy`
- `gif`
- `vtk`

Video output remains supported in the writer layer but is not enabled by the
default random-run output set.

## Architecture

- **PDEs** (`plm_data/pdes/`) expose PDE specs and runtime factories for the
  migrated random-run PDE set.
- **Runtime config** (`plm_data/core/runtime_config.py`) contains the concrete
  in-memory dataclasses passed to PDE runtimes. YAML configs and shared
  fragments are not part of the refactored architecture.
- **Sampling** (`plm_data/sampling/`) chooses a compatible random PDE/domain
  case, boundary scenario, initial-condition scenario, parameters, solver
  profile, time settings, and output settings from a required seed.
- **Domains** (`plm_data/domains/`) provide Gmsh-backed domain specs, mesh
  builders, validation, and domain sampling used by random PDE profiles.
- **Boundary-condition operators and scenarios**
  (`plm_data/boundary_conditions/`) separate low-level operator behavior from
  PDE-level scenario generation.
- **Initial-condition operators and scenarios**
  (`plm_data/initial_conditions/`) separate field construction behavior from
  complete PDE-level initialization choices.
- **Field expression runtime** (`plm_data/fields/`) contains shared
  scalar/vector expression resolution, UFL rendering, interpolation, and source
  term helpers.
- **Stochastic runtime** (`plm_data/stochastic/`) contains deterministic
  cell-noise, state forcing, and randomized coefficient helpers.
- **Output** (`plm_data/core/output.py`) validates requested fields, expands
  vector outputs into grid components, writes numpy/GIF/VTK/video formats, and
  records post-run diagnostics in `frames_meta.json` and `run_meta.json`.

The runner owns simulation generation directly; curated YAML config files and
the old config loader have been removed from the runtime path.

## Tests

Run tests from the conda environment:

```bash
python -m pytest tests/
python -m ruff check plm_data tests
python -m pyright plm_data
```

Pytest is configured to use four workers by default. For local debugging, pass
`-n 0`.

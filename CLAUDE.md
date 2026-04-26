This file provides guidance to coding agents working in this repository.

## What This Is

PLM-data generates random two-dimensional, time-dependent PDE simulation
datasets using DOLFINx (FEniCSx), Gmsh, and grid/media output writers. The
project optimizes for breadth of sampled dynamics and simple operation, not for
being a general-purpose PDE solver framework. More details on the mission and design are in the [Spec](SPEC.md).

## Commands

Make sure the conda environment `fenicsx-env` is activated before running commands.

```bash
# Run one random simulation. A seed is required.
./run.sh --seed 1234

# Run on four pinned physical cores.
./run.sh -n 4 --seed 1234

# Reserve disjoint physical cores for concurrent simulations.
./run.sh -n 4 --core-list 0-3 --seed 1234
./run.sh -n 4 --core-list 4-7 --seed 5678
```

`run.sh -n N` means "use N physical cores": one single-threaded MPI rank per
physical core, pinned with MPI binding. Use `--core-list` when multiple
simulations should coexist on disjoint core sets without oversubscription.

Tests run via `python -m pytest tests/`. Pytest is configured to use 4 parallel
processes (`-n 4` via `pytest-xdist`) by default. Use `-n 0` for local debugging.

Run style/type checks after code changes:

```bash
python -m ruff check plm_data tests
python -m ruff format plm_data tests
python -m pyright plm_data
```

## Architecture

The active random-run architecture is built around first-class objects:

1. **PDEs** (`plm_data/pdes/`) - PDE specs and runtime factories for the
   migrated random-run PDE set. Random-run PDE limits, including solver choice,
   time settings, output settings, domain constraints, and randomized
   coefficients, are attached to PDE specs via `random_options`.
2. **Runtime config** (`plm_data/core/runtime_config.py`) - concrete in-memory
   dataclasses used by the sampler, runner, output writers, and PDE runtimes.
3. **Sampling** (`plm_data/sampling/`) - seed-based random runtime-config
   generation, compatibility checks, bounded retry, and run identity metadata.
4. **Domains** (`plm_data/domains/`) - Gmsh-backed 2D domain specs, mesh
   builders, validation, and spec-owned random domain profiles used by the
   sampler.
5. **Boundary-condition operators and scenarios**
   (`plm_data/boundary_conditions/`) - low-level operator specs remain under
   `operators/`; complete PDE-level random choices live under `scenarios/`.
6. **Initial-condition operators and scenarios**
   (`plm_data/initial_conditions/`) - low-level field operators remain
   available; complete PDE-level random choices live under `scenarios/`.
7. **Field expression runtime** (`plm_data/fields/`) - shared scalar/vector
   field expression resolution, UFL rendering, interpolation, and source-term
   helpers used by PDEs, BCs, ICs, and stochastic coefficients.
8. **Stochastic runtime** (`plm_data/stochastic/`) - deterministic cell-noise,
   dynamic state forcing, and randomized coefficient construction.
9. **Output** (`plm_data/core/output.py`) - validates output fields, expands
   vector outputs, writes numpy/GIF/VTK/video outputs, and records diagnostics.

Curated YAML configs and the old config loader have been removed. The runner
owns in-memory random simulation generation directly.
Random sampling is spec-driven: the sampler intersects PDE random options,
domain random profiles, and BC/IC scenario compatibility rather than carrying
per-PDE runtime builders or hard-coded domain cases.

## Output

Random runs write to:

```text
output/<pde>/<domain>/seed_<seed>_<run-id>/
```

Default random-run formats are numpy arrays, GIFs, and VTK/ParaView files.
Successful runs write `run_meta.json`, `frames_meta.json`, field arrays/media,
and a `paraview/` directory when VTK output is enabled.

When creating or migrating PDEs, keep the working 2D output so the developer can
visually verify the PDE is working.

## Coding Rules

- Do not use parameter defaults in function signatures. We prefer explicitness.
- If asked to work in a new git worktree, create it in the `./.worktrees/` directory and name it after the PDE or feature being developed.
- After merging worktrees, delete them and their branches to avoid confusion.

- Use ruff for linting and formatting.
- Check pyright after a logical batch of edits.
- Do not use `from __future__ import annotations`.

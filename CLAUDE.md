This file provides guidance to all coding agents when working with code in this repository.

## What This Is

PLM-data generates PDE simulation datasets using DOLFINx (FEniCSx). It produces numpy arrays on regular grids from finite-element simulations, intended for training ML models. The project is inspired by and references the [VisualPDE](https://visualpde.com/) simulation catalog.

## Commands

```bash
# Run a simulation (single core)
./run.sh run configs/basic/heat/2d_default.yaml

# Run a simulation with multiple MPI ranks
./run.sh -n 4 run configs/basic/heat/2d_default.yaml

# List registered presets
./run.sh list
```

Tests run via `python -m pytest tests/`. The project runs directly as a Python module. DOLFINx and its dependencies (PETSc, mpi4py, UFL) must be installed in the environment.

## Architecture

The system has three layers:

1. **Presets** (`plm_data/presets/`) — Each PDE is a self-contained class registered via `@register_preset("name")`. Presets expose:
   - `spec` — a `PresetSpec` describing parameters, config-facing inputs, solved states, selectable outputs, and supported dimensions
   - `build_problem(config)` — returns a runtime problem object for one of the shared engines or for the custom escape hatch
   Common solver families live in `plm_data/presets/base.py` as `StationaryLinearProblem`, `TransientLinearProblem`, `TransientNonlinearProblem`, and `CustomProblem`.

2. **Core** (`plm_data/core/`) — Shared infrastructure:
   - `config.py` — `SimulationConfig` dataclass loaded from YAML. Configs are validated against the preset spec up front. The schema is input-centric: each entry under `inputs` owns explicit `boundary_conditions`, `source`, and `initial_condition` settings, while `output.fields` selects which declared outputs are saved. Transient presets use a `time:` section. Output resolution lives under `output.resolution`. Domains must declare `periodic_axes` explicitly, even when empty, and config-facing boundary conditions may not target periodic faces.
   - `mesh.py` — `create_domain()` returns `DomainGeometry` (mesh + facet_tags + boundary_names + ds measure + axis bounds / periodic metadata). Domain creation is registry-backed. Built-in domains auto-tag boundaries (x-, x+, y-, y+ for rectangle; 6 faces for box; x-/x+ for interval)
   - `periodic.py` — shared `dolfinx_mpc` integration for domain-level periodic constraints. Presets should treat periodicity as a domain property via `domain.periodic_axes`, not as a boundary-condition type.
   - `spatial_fields.py` — Shared scalar and vector spatial field system (constant, sine_product, gaussian_bump, step, none, custom). Provides UFL builders, scalar interpolators, and vector-component expansion helpers. Supports `"param:name"` references
  - `boundary_conditions.py` — shared scalar Dirichlet / Neumann / Robin helpers plus vector Dirichlet / Neumann helpers for standard vector-valued spaces; vector Robin remains intentionally unsupported in the shared layer
   - `source_terms.py` — scalar and vector source-form builders from the unified field expression config
   - `initial_conditions.py` — scalar and vector IC helpers from the unified field expression config; `random_perturbation` stays scalar-only and DOF-based
   - `runner.py` — `SimulationRunner` orchestrates: loads config → instantiates preset → builds problem → runs it → finalizes output
   - `output.py` — `FrameWriter` coordinates format-specific writers. Validates base fields against the preset spec, expands vector fields into component outputs for grid-based formats. Delegates to writers in `formats/`
   - `formats/` — Output format writers: `NumpyWriter` (.npy arrays), `GifWriter` (animated .gif), `VideoWriter` (.mp4), `VTKWriter` (pyvista .vtu/.pvd for Paraview). Grid writers (numpy/gif/video) share the interpolation pipeline; VTK writes FEM functions directly
   - `interpolation.py` — `function_to_array()` maps DOLFINx FEM functions onto regular numpy grids via point evaluation

3. **Configs** (`configs/<category>/<preset>/`) — YAML files specifying: preset name, physical parameters, domain geometry, optional `time`, explicit per-input config blocks under `inputs`, solver options, output settings, and seed.
   The current schema uses explicit per-input blocks under `inputs` plus output selection under `output.fields`.

## Adding a New PDE Preset

1. Create `plm_data/presets/<category>/<name>.py`
2. Implement `spec` with a `PresetSpec`
3. Implement `build_problem(config)` returning one of the shared problem engines or a `CustomProblem`
4. Decorate with `@register_preset("name")`
5. Create a YAML config in `configs/<category>/<name>/`
6. No central import list is needed; preset modules are auto-discovered recursively

## Reference Material

`reference/pdes/` contains markdown descriptions of PDE simulations from VisualPDE, organized by category (_basic-pdes, _fluids, _mathematical-biology, _nonlinear-physics). Use the `/reference` skill to look up parameters and descriptions when implementing or tuning presets.

## Key Conventions

- YAML configs must be fully explicit — no hidden defaults in code
- Domain periodicity is configured explicitly with `domain.periodic_axes`; non-periodic configs still declare `[]`
- Config validation is spec-driven: parameter names, input names, output names, allowed sections, output modes, and supported dimensions are checked before solving
- Output goes to `output/<category>/<preset>/` with format-specific files: `.npy` arrays, `.gif`/`.mp4` animations, and `paraview/` directory with `.pvd`+`.vtu` files for Paraview
- Presets are auto-discovered recursively under `plm_data.presets`
- Meshes use `GhostMode.shared_facet` for DOLFINx compatibility
- PETSc solver option prefixes follow the pattern `plm_` or `plm_<preset>_`

## Coding Rules

- Always! think about the optimal, cleanest way to implement a feature. Is the current code structure the best way to support this, or is there a more elegant design? Refactor if needed.
- Don't account for backwards compatibility if you are planning code changes. This is a research codebase, not a production library.
- Do not use parameter defaults in code. The config must specify all parameters explicitly.
- use ruff for linting and formatting. Make sure to run ruff after making changes to ensure code style consistency.
- After making code changes, check Pylance diagnostics using `mcp__ide__getDiagnostics` to catch type errors and other issues. Do this after completing a logical batch of edits, not after every single edit.
- Do not use `from __future__ import annotations`. This is a Python 3.10+ codebase, and we want to keep type annotations straightforward without string literals.

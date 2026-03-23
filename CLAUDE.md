# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

PLM-data generates PDE simulation datasets using DOLFINx (FEniCSx). It produces numpy arrays on regular grids from finite-element simulations, intended for training ML models. The project is inspired by and references the [VisualPDE](https://visualpde.com/) simulation catalog.

## Commands

```bash
# Run a simulation from a YAML config
python -m plm_data run configs/basic/heat/2d_default.yaml

# List registered presets
python -m plm_data list
```

Tests run via `python -m pytest tests/`. The project runs directly as a Python module. DOLFINx and its dependencies (PETSc, mpi4py, UFL) must be installed in the environment.

## Architecture

The system has three layers:

1. **Presets** (`plm_data/presets/`) — Each PDE is a self-contained class registered via `@register_preset("name")`. Presets own their full solve logic (mesh, function space, variational form, solver, time-stepping). Three base classes:
   - `PDEPreset` — abstract base, implement `run()` directly
   - `SteadyLinearPreset` — for steady-state linear problems; implement `create_function_space()`, `create_boundary_conditions()`, `create_forms()`; methods receive `DomainGeometry` (not bare mesh); the base handles domain creation, `LinearProblem` solve, and frame output
   - `TimeDependentPreset` — for time-dependent problems; implement `setup()`, `step()`, `get_output_fields()`; the base handles the time loop and output scheduling

2. **Core** (`plm_data/core/`) — Shared infrastructure:
   - `config.py` — `SimulationConfig` dataclass loaded from YAML; all fields explicit, no hidden defaults. BCs, source terms, and ICs are all per-field (keyed by field name like "u", "velocity", "pressure")
   - `mesh.py` — `create_domain()` returns `DomainGeometry` (mesh + facet_tags + boundary_names + ds measure). Built-in domains auto-tag boundaries (x-, x+, y-, y+ for rectangle; 6 faces for box). Future Gmsh support will populate from physical groups
   - `spatial_fields.py` — Shared spatial field type system (constant, sine_product, gaussian_bump, step, none, custom). Two renderers: `build_ufl_field()` for variational forms, `build_interpolator()` for numpy interpolation. Supports `"param:name"` references. This is the central hub — BCs, source terms, and ICs all delegate to it
   - `boundary_conditions.py` — `apply_dirichlet_bcs()` creates DirichletBC objects; `build_natural_bc_forms()` returns (a_bc, L_bc) for Neumann and Robin BCs
   - `source_terms.py` — `build_source_form()` constructs f*v*dx for scalar fields; `build_vector_source_form()` assembles per-component scalar source terms into a vector body force for vector PDEs
   - `initial_conditions.py` — `apply_ic(func, ic_config, parameters, seed)` delegates spatial types to `build_interpolator()` from spatial_fields; handles `random_perturbation` directly (DOF-based). `apply_vector_ic()` applies per-component scalar ICs to vector functions for vector PDEs
   - `runner.py` — `SimulationRunner` orchestrates: loads config → instantiates preset → calls `preset.run()` → finalizes output
   - `output.py` — `FrameWriter` accumulates field snapshots in memory, then saves one `(num_frames, *resolution)` array per field in `finalize()`
   - `interpolation.py` — `function_to_array()` maps DOLFINx FEM functions onto regular numpy grids via point evaluation

3. **Configs** (`configs/<category>/<preset>/`) — YAML files specifying: preset name, physical parameters, domain (pure geometry), per-field boundary conditions (dirichlet/neumann/robin), per-field source terms, per-field initial conditions, output resolution, time-stepping (dt, t_end), solver options, and seed.

## Adding a New PDE Preset

1. Create `plm_data/presets/<category>/<name>.py`
2. Subclass `SteadyLinearPreset` or `TimeDependentPreset` (or `PDEPreset` for full control)
3. Decorate with `@register_preset("name")`
4. Provide a `metadata` property returning `PDEMetadata`
5. Import the new module in `plm_data/presets/<category>/__init__.py`
6. If adding a new category, also import it in `plm_data/presets/__init__.py` inside `_load_all_presets()`
7. Create a YAML config in `configs/<category>/<name>/`

## Reference Material

`reference/pdes/` contains markdown descriptions of PDE simulations from VisualPDE, organized by category (_basic-pdes, _fluids, _mathematical-biology, _nonlinear-physics). Use the `/reference` skill to look up parameters and descriptions when implementing or tuning presets.

## Key Conventions

- YAML configs must be fully explicit — no hidden defaults in code
- Output goes to `output/<category>/<preset>/<field>.npy` as a single `(num_frames, *resolution)` array
- Presets are auto-discovered via module imports triggered by `_load_all_presets()`
- Meshes use `GhostMode.shared_facet` for DOLFINx compatibility
- PETSc solver option prefixes follow the pattern `plm_` or `plm_<preset>_`

## Coding Rules

- Always! think about the optimal, cleanest way to implement a feature. Is the current code structure the best way to support this, or is there a more elegant design? Refactor if needed.
- Don't account for backwards compatibility if you are planning code changes. This is a research codebase, not a production library.
- Do not use parameter defaults in code. The config must specify all parameters explicitly.
- use ruff for linting and formatting. Make sure to run ruff after making changes to ensure code style consistency.
- After making code changes, check Pylance diagnostics using `mcp__ide__getDiagnostics` to catch type errors and other issues. Do this after completing a logical batch of edits, not after every single edit.
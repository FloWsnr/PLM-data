This file provides guidance to all coding agents when working with code in this repository.

## What This Is

PLM-data generates PDE simulation datasets using DOLFINx (FEniCSx). It produces numpy arrays on regular grids from finite-element simulations, intended for training ML models. The project is inspired by and references the [VisualPDE](https://visualpde.com/) simulation catalog.

## Commands

```bash
# Run a simulation (single core)
./run.sh run configs/basic/heat/2d_default.yaml --output-dir ./output

# Run a simulation with multiple MPI ranks
./run.sh -n 4 run configs/basic/heat/2d_default.yaml --output-dir ./output

# List registered presets
./run.sh list
```

Tests run via `python -m pytest tests/`. pytest is configured to use 4 parallel processes (`-n 4` via `pytest-xdist`) by default. The project runs directly as a Python module. DOLFINx and its dependencies (PETSc, mpi4py, UFL) must be installed in the environment.

## Conda Environments

Two conda environments exist:
- `fenicsx-env` — real-valued PETSc (default). Used by all presets except `maxwell`.
- `fenicsx-env-complex` — complex-valued PETSc (`petsc=*=complex*`). Required by the `maxwell` preset. Does not have `dolfinx_mpc`.

Override the environment via `PLM_CONDA_ENV`:
```bash
PLM_CONDA_ENV=fenicsx-env-complex ./run.sh -n 4 run configs/physics/maxwell/2d_default.yaml --output-dir ./output
```

## Architecture

The system has three layers:

1. **Presets** (`plm_data/presets/`) — Each PDE is a self-contained class registered via `@register_preset("name")`. Presets expose:
   - `spec` — a `PresetSpec` describing parameters, config-facing coefficients/inputs, boundary-condition fields/operators, solved states, selectable outputs, `static_fields` excluded from stagnation warnings, and supported dimensions
   - `build_problem(config)` — returns a runtime problem object for one of the shared engines or for the custom escape hatch
   Common solver families live in `plm_data/presets/base.py` as `StationaryLinearProblem`, `TransientLinearProblem`, `TransientNonlinearProblem`, and `CustomProblem`.
   Family-specific helpers can live alongside presets when multiple presets share a discretization; for example, `plm_data/presets/fluids/_taylor_hood.py` is the shared Stokes / Navier-Stokes incompressible-flow helper.

2. **Core** (`plm_data/core/`) — Shared infrastructure:
   - `config.py` — `SimulationConfig` dataclass loaded from YAML. Configs are validated against the preset spec up front. The schema uses explicit top-level `coefficients`, `inputs`, `boundary_conditions`, and `output` sections. Coefficients are preset-declared field expressions used directly in forms; inputs own `source` and `initial_condition`; boundary conditions are declared separately per preset boundary field; `output.fields` selects which declared outputs are saved. Transient presets use a `time:` section. Output resolution lives under `output.resolution`, while the output root directory is provided at runtime via `--output-dir`. Domains may declare extra `periodic_maps`, while built-in domains provide the standard face-pair maps used by periodic boundary operators.
   - `mesh.py` — `create_domain()` returns `DomainGeometry` (mesh + facet_tags + boundary_names + ds measure + axis bounds / periodic metadata). Domain creation is registry-backed. Built-in domains auto-tag boundaries (x-, x+, y-, y+ for rectangle; 6 faces for box; x-/x+ for interval)
   - `periodic.py` — shared `dolfinx_mpc` integration for periodic boundary operators. Periodicity is activated per boundary field via the `periodic` operator and resolved against the domain's available periodic maps.
   - `spatial_fields.py` — Shared scalar and vector spatial field system (constant, sine_product, cosine_product, gaussian_bump, radial_cosine, step, none, custom). Provides UFL builders, scalar interpolators, and vector-component expansion helpers. Supports `"param:name"` references
  - `boundary_conditions.py` — shared scalar Dirichlet / Neumann / Robin helpers plus vector Dirichlet / Neumann helpers for standard vector-valued spaces; vector Robin remains intentionally unsupported in the shared layer
   - `source_terms.py` — scalar and vector source-form builders from the unified field expression config
   - `initial_conditions.py` — scalar and vector IC helpers from the unified field expression config; `random_perturbation` stays scalar-only and DOF-based
   - `runner.py` — `SimulationRunner` orchestrates: loads config → instantiates preset → builds problem → runs it → finalizes output
   - `output.py` — `FrameWriter` coordinates format-specific writers. It validates base fields against the preset spec, expands vector fields into component outputs for grid-based formats, and writes post-run stagnation diagnostics to `frames_meta.json`, skipping outputs listed in `spec.static_fields`. Delegates to writers in `formats/`
   - `formats/` — Output format writers: `NumpyWriter` (.npy arrays), `GifWriter` (animated .gif), `VideoWriter` (.mp4), `VTKWriter` (pyvista .vtu/.pvd for Paraview). Grid writers (numpy/gif/video) share the interpolation pipeline; VTK writes FEM functions directly
   - `interpolation.py` — `function_to_array()` maps DOLFINx FEM functions onto regular numpy grids via point evaluation

3. **Configs** (`configs/<category>/<preset>/`) — YAML files specifying: preset name, physical parameters, explicit coefficients, domain geometry, explicit `inputs`, explicit `boundary_conditions`, optional `time`, solver strategy/profile settings, output settings, and seed.
   The current schema uses top-level `coefficients`, `inputs`, `boundary_conditions`, `output.fields`, and an explicit `solver` block with `strategy`, `serial`, and `mpi`. Shared YAML fragments live in `configs/_fragments.yaml` and can be reused anywhere via `$ref`.

## Adding a New PDE Preset

1. Create `plm_data/presets/<category>/<name>.py`
2. Implement `spec` with a `PresetSpec`, including explicit `boundary_fields` and `static_fields`
3. Implement `build_problem(config)` returning one of the shared problem engines or a `CustomProblem`
4. Decorate with `@register_preset("name")`
5. Create a YAML config in `configs/<category>/<name>/`
6. No central import list is needed; preset modules are auto-discovered recursively

## Reference Material

`reference/pdes/` contains markdown descriptions of PDE simulations from VisualPDE, organized by category (_basic-pdes, _fluids, _mathematical-biology, _nonlinear-physics). Use the `/reference` skill to look up parameters and descriptions when implementing or tuning presets.

## Key Conventions

- YAML configs must be fully explicit — no hidden defaults in code
- Shared config fragments are allowed via `$ref: some.path` into `configs/_fragments.yaml`; mappings may override referenced mappings locally, but reuse still stays explicit in YAML
- Solver configs must declare `solver.strategy`, `solver.serial`, and `solver.mpi` explicitly; the runtime selects the serial or MPI profile from communicator size
- Periodic constraints are declared with `boundary_conditions.<field>.<side>.[].operator: periodic`; optional `domain.periodic_maps` can add custom geometric pairings beyond the built-in domain maps
- Config validation is spec-driven: parameter names, input names, boundary field names, output names, allowed sections, boundary operators, output modes, and supported dimensions are checked before solving
- Output goes to `output/<category>/<preset>/` with format-specific files: `.npy` arrays, `.gif`/`.mp4` animations, and `paraview/` directory with `.pvd`+`.vtu` files for Paraview
- Presets must declare `static_fields` explicitly; post-run stagnation warnings skip those outputs
- Presets are auto-discovered recursively under `plm_data.presets`
- Meshes use `GhostMode.shared_facet` for DOLFINx compatibility
- PETSc solver option prefixes follow the pattern `plm_` or `plm_<preset>_`

## Coding Rules

- Always! think about the optimal, cleanest way to implement a feature. Is the current code structure the best way to support this, or is there a more elegant design? Refactor if needed.
- Don't account for backwards compatibility if you are planning code changes. This is a research codebase, not a production library.
- Do not use parameter defaults in code. The config must specify all parameters explicitly.
- use ruff for linting and formatting. Make sure to run ruff after making changes to ensure code style consistency.
- After making code changes, check pyright or the Pylance MCP to catch type errors and other issues. Do this after completing a logical batch of edits, not after every single edit.
- Do not use `from __future__ import annotations`. This is a Python 3.10+ codebase, and we want to keep type annotations straightforward without string literals.

## Simulation Rules

- Use 4 CPUs for all simulations runs by default to speed up data generation.
- use the `./output` directory for all simulation outputs
- create a new subdirectory under `./output` for each new simulation run, following the pattern `./output/<category>/<preset>/`
- Delete old simulations if they are no longer needed to avoid cluttering the output directory

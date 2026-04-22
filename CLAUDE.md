This file provides guidance to all coding agents when working with code in this repository.

## What This Is

PLM-data generates PDE simulation datasets using DOLFINx (FEniCSx) and custom structured-grid solvers. It produces numpy arrays on regular grids from finite-element simulations and structured-grid PDE solvers, intended for training ML models. The project is inspired by and references the [VisualPDE](https://visualpde.com/) simulation catalog.

## Commands

```bash
# Run a simulation (single core)
./run.sh run configs/basic/heat/2d_localized_blob_diffusion.yaml --output-dir ./output

# Run a simulation on 4 pinned physical cores
./run.sh -n 4 run configs/basic/heat/2d_localized_blob_diffusion.yaml --output-dir ./output

# Run one config many times with incrementing seeds from the YAML config
./run.sh -n 4 run configs/basic/heat/2d_localized_blob_diffusion.yaml --output-dir ./output --n-runs 100

# Reserve disjoint physical cores for concurrent simulations
./run.sh -n 4 --core-list 0-3 run configs/basic/heat/2d_localized_blob_diffusion.yaml --output-dir ./output
./run.sh -n 4 --core-list 4-7 run configs/basic/heat/2d_localized_blob_diffusion.yaml --output-dir ./output --n-runs 100

# List registered presets
./run.sh list

# Build one HTML gallery from all PDE GIFs under a directory
./run.sh gallery ./output
```

`run.sh -n N` now means "use N physical cores": one single-threaded MPI rank per physical core, pinned with MPI binding. Use `--core-list` when multiple simulations should coexist on disjoint core sets without oversubscription.

Tests run via `python -m pytest tests/`. pytest is configured to use 4 parallel processes (`-n 4` via `pytest-xdist`) by default. The project runs directly as a Python module. DOLFINx and its dependencies (PETSc, mpi4py, UFL) must be installed in the environment.

## Conda Environments

Two conda environments exist:
- `fenicsx-env` — real-valued PETSc (default). Used by the shipped presets, including `maxwell_pulse`.
- `fenicsx-env-complex` — complex-valued PETSc (`petsc=*=complex*`). Optional for separate experiments. Does not have `dolfinx_mpc`.

Override the environment via `PLM_CONDA_ENV`:
```bash
PLM_CONDA_ENV=fenicsx-env-complex ./run.sh -n 4 run configs/basic/heat/2d_localized_blob_diffusion.yaml --output-dir ./output
```

## Architecture

The system has three layers:

1. **Presets** (`plm_data/presets/`) — Each PDE is a self-contained class registered via `@register_preset("name")`. Presets expose:
   - `spec` — a `PresetSpec` describing parameters, config-facing coefficients/inputs, boundary-condition fields/operators, solved states, selectable outputs, `static_fields` excluded from stagnation warnings, and supported dimensions
   - `build_problem(config)` — returns a runtime problem object for one of the shared engines or for the custom escape hatch
   Common solver families live in `plm_data/presets/base.py` as `TransientLinearProblem`, `TransientNonlinearProblem`, and `CustomProblem`.
   Family-specific helpers can live alongside presets when multiple presets share a backend or discretization.

2. **Core** (`plm_data/core/`) — Shared infrastructure:
   - `config.py` — `SimulationConfig` dataclass loaded from YAML. `load_config()` is declarative: it parses and validates sampler specs, parameter references, and the preset schema, but it does not turn sampled values into concrete runtime numbers. The schema uses explicit top-level `coefficients`, `inputs`, `boundary_conditions`, and `output` sections. Coefficients are preset-declared field expressions used directly in forms; inputs own `source` and `initial_condition`; boundary conditions are declared separately per preset boundary field; `output.fields` selects which declared outputs are saved. Transient presets use a `time:` section. Output resolution lives under `output.resolution`, while the output root directory is provided at runtime via `--output-dir`. Single runs overwrite that preset directory after an initial cleanup; `--n-runs` uses the config seed as the base seed and writes each run into `seed_<seed>` subdirectories below the preset directory. Domains may declare extra `periodic_maps`, while built-in domains provide the standard face-pair maps used by periodic boundary operators.
   - `config_realization.py` — turns a declarative `SimulationConfig` into one concrete seeded run configuration. Realization applies shared sampler syntax across top-level parameters, domains, coefficients, sources, initial conditions, boundary values, and boundary operator parameters. Domain sampling exists for built-in domains but stays opt-in via `domain.allow_sampling: true`.
   - `mesh.py` — `create_domain()` returns `DomainGeometry` (mesh + facet_tags + boundary_names + ds measure + periodic metadata). Domain creation is registry-backed. The current repo ships `interval`, `rectangle`, `box`, `disk`, `dumbbell`, `l_shape`, `multi_hole_plate`, `parallelogram`, `channel_obstacle`, `y_bifurcation`, `venturi_channel`, `porous_channel`, `serpentine_channel`, `airfoil_channel`, `side_cavity_channel`, and `annulus`. Cartesian/parallelogram domains expose the standard face tags and periodic maps; Gmsh-backed domains require `python-gmsh`, with `annulus` tagged as `inner`/`outer`, `channel_obstacle` tagged as `inlet`/`outlet`/`walls`/`obstacle`, `l_shape` tagged as `outer`/`notch`, `multi_hole_plate` tagged as `outer` plus either grouped `holes` or explicit `hole_*` names, `y_bifurcation` tagged as `inlet`/`outlet_upper`/`outlet_lower`/`walls`, and `venturi_channel`, `porous_channel`, `serpentine_channel`, `airfoil_channel`, and `side_cavity_channel` tagged with domain-specific named boundaries defined by their respective factories.
   - `periodic.py` — shared `dolfinx_mpc` integration for periodic boundary operators. Periodicity is activated per boundary field via the `periodic` operator and resolved against the domain's available periodic maps.
   - `spatial_fields.py` — Shared scalar and vector spatial field system (constant, sine_waves, gaussian_bump, radial_cosine, step, none, custom). Provides UFL builders, scalar interpolators, and vector-component expansion helpers. Supports `"param:name"` references
  - `boundary_conditions.py` — shared scalar Dirichlet / Neumann / Robin helpers plus vector Dirichlet / Neumann helpers for standard vector-valued spaces; vector Robin remains intentionally unsupported in the shared layer
   - `source_terms.py` — scalar and vector source-form builders from the unified field expression config
   - `initial_conditions.py` — scalar and vector IC helpers from the unified field expression config, including sampled IC parameters and the shared `gaussian_noise`, `gaussian_blobs`, `sine_waves`, `quadrants`, and `constant` initial-condition families
   - `runner.py` — `SimulationRunner` orchestrates: loads config → realizes sampled values for one seed → instantiates preset → builds problem → runs it → finalizes output
   - `output.py` — `FrameWriter` coordinates format-specific writers. It validates base fields against the preset spec, expands vector fields into component outputs for grid-based formats, and writes post-run stagnation diagnostics to `frames_meta.json`, skipping outputs listed in `spec.static_fields`. Delegates to writers in `formats/`
   - `formats/` — Output format writers: `NumpyWriter` (.npy arrays), `GifWriter` (animated .gif), `VideoWriter` (.mp4), `VTKWriter` (native DOLFINx `.pvd`/`.vtu` output for ParaView). Grid writers (numpy/gif/video) share the interpolation pipeline; VTK writes FEM functions directly
   - `interpolation.py` — `function_to_array()` maps DOLFINx FEM functions onto regular numpy grids via point evaluation. Grid points outside the mesh (e.g. corners/holes on non-rectangular domains) produce NaN; `InterpolationCache.outside_mask` tracks which points are out-of-domain. When a domain mask is present, `FrameWriter` writes `domain_mask.npy` (True = inside) alongside the field arrays.
   - `plm_data/tools/gif_gallery.py` — utility for postprocessing GIF outputs into a single HTML gallery. It scans a directory recursively for `.gif` files, groups them by containing run directory, and renders one PDE per row with one column per field.

3. **Configs** (`configs/<category>/<preset>/`) — YAML files specifying: preset name, physical parameters, optional shared sampled parameters referenced elsewhere via `param:...`, explicit coefficients, domain geometry, explicit `inputs`, explicit `boundary_conditions`, optional `time`, solver strategy/profile settings, output settings, and seed.
   The current schema uses top-level `coefficients`, `inputs`, `boundary_conditions`, `output.fields`, and an explicit `solver` block with `strategy`, `serial`, and `mpi`. Shared YAML fragments live in `configs/_fragments.yaml` and can be reused anywhere via `$ref`.

## Adding a New PDE Preset

1. Create `plm_data/presets/<category>/<name>.py`
2. Implement `spec` with a `PresetSpec`, including explicit `boundary_fields` and `static_fields`
3. Implement `build_problem(config)` returning one of the shared problem engines or a `CustomProblem`
4. Decorate with `@register_preset("name")`
5. Create a YAML config in `configs/<category>/<name>/`
6. No central import list is needed; preset modules are auto-discovered recursively

## Reference Material

`reference/pdes/` contains markdown descriptions of PDE simulations from VisualPDE, organized by category (_basic-pdes, _fluids, _mathematical-biology, _nonlinear-physics). Use that local reference material directly when implementing or tuning presets; if your agent environment exposes a reference lookup skill, use the appropriate one for that environment.

## Key Conventions

- YAML configs must be fully explicit — no hidden defaults in code
- Shared config fragments are allowed via `$ref: some.path` into `configs/_fragments.yaml`; mappings may override referenced mappings locally, but reuse still stays explicit in YAML
- Solver configs must declare `solver.strategy`, `solver.serial`, and `solver.mpi` explicitly; the runtime selects the serial or MPI profile from communicator size
- Periodic constraints are declared with `boundary_conditions.<field>.<side>.[].operator: periodic`; optional `domain.periodic_maps` can add custom geometric pairings beyond the built-in domain maps
- Config validation is spec-driven: preset-declared parameter names, input names, boundary field names, output names, allowed sections, boundary operators, output modes, and supported dimensions are checked before solving; additional top-level parameters are allowed when used as explicit shared sampled values via `param:...`
- Config loading and runtime realization are separate steps: `load_config()` returns the declarative config, and `realize_simulation_config()` produces the concrete per-seed runtime config used by the solver
- Output goes to `output/<category>/<preset>/` for single runs, which clear that directory before writing format-specific files: `.npy` arrays, `.gif`/`.mp4` animations, and `paraview/` directory with `.pvd`+`.vtu` files for Paraview
- Batch runs use `--n-runs` and write to `output/<category>/<preset>/seed_<seed>/`, incrementing from the config seed
- The `gallery` CLI command scans those output trees recursively and writes a `pde_gif_gallery.html` page that aligns PDE runs by row and field names by column
- Presets must declare `static_fields` explicitly; post-run stagnation warnings skip those outputs
- Presets are auto-discovered recursively under `plm_data.presets`
- Meshes use `GhostMode.shared_facet` for DOLFINx compatibility
- PETSc solver option prefixes follow the pattern `plm_` or `plm_<preset>_`

## Coding Rules

- Always use git worktrees to isolate different features or bug fixes into separate branches. This keeps the commit history clean and makes it easier to review changes. Remove the worktrees when the feature or fix is merged to avoid clutter.
- Always! think about the optimal, cleanest way to implement a feature. Is the current code structure the best way to support this, or is there a more elegant design? Refactor if needed.
- Don't account for backwards compatibility if you are planning code changes. This is a research codebase, not a production library.
- Do not use parameter defaults in code. The config must specify all parameters explicitly.
- use ruff for linting and formatting. Make sure to run ruff after making changes to ensure code style consistency.
- After making code changes, check pyright or the Pylance MCP to catch type errors and other issues. Do this after completing a logical batch of edits, not after every single edit.
- Do not use `from __future__ import annotations`. This is a Python 3.10+ codebase, and we want to keep type annotations straightforward without string literals.

## Simulation Rules

- Use 4 physical cores for simulation runs by default to speed up data generation. `./run.sh -n 4` now launches 4 pinned single-threaded MPI ranks; add `--core-list` to reserve specific physical cores when running multiple simulations concurrently.
- use the `./output` directory for all simulation outputs
- single runs overwrite `./output/<category>/<preset>/` after cleaning it first
- dataset-generation batches should use `--n-runs`, which creates `./output/<category>/<preset>/seed_<seed>/` subdirectories with consecutive seeds from the config
- Delete old simulations if they are no longer needed to avoid cluttering the output directory. However, when creating new PDE presets, you need to keep the working 2D output so the developer can visually verify the new preset is working correctly.

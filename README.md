# PLM-data

Generate PDE simulation datasets using [DOLFINx](https://github.com/FEniCS/dolfinx) (FEniCSx) and openfoam.

## Installation

### DOLFINx (FEniCSx)

Create a conda environment with DOLFINx and its dependencies:

```bash
conda create -n fenicsx-env -c conda-forge fenics-dolfinx mpich python=3.12 ffmpeg python-gmsh
conda activate fenicsx-env
conda install -c conda-forge dolfinx_mpc
pip install pyyaml pytest pytest-xdist pyright matplotlib
```

See the [DOLFINx README](https://github.com/FEniCS/dolfinx#installation) for alternative installation methods (Docker, apt, Spack, Windows).

### Openfoam

Either install openfoam (https://openfoam.org/download/13-ubuntu/) or load it as a module if available on your HPC.

## Architecture

- **Presets** (`plm_data/presets/`) — Each PDE is a `PDEPreset` with a `PresetSpec` and a `build_problem(config)` factory. Specs now separate config-facing `coefficients` and `inputs`, boundary-condition fields/operators, solved `states`, selectable `outputs`, and explicit `static_fields` that are excluded from post-run stagnation warnings.
- **Problem engines** (`plm_data/presets/base.py`) — Reusable runtime engines cover transient linear, transient nonlinear, and custom problems. Presets still own the formulation details and use shared runtime loops where that helps.
- **Config** (`configs/`) — YAML uses explicit top-level `coefficients:`, `inputs:`, `boundary_conditions:`, and `output:` sections. Coefficients are preset-declared field expressions used directly in variational forms. Inputs configure sources and initial conditions; `boundary_conditions` configures each preset boundary field; `output.fields` selects which declared outputs to save and how to expand them. Presets use a `time:` section; output resolution lives under `output.resolution`. The output root directory is chosen at the CLI with `--output-dir`; single runs clean and write into `<output-dir>/<category>/<preset>`, while batch runs use `--n-runs N` and write into `<output-dir>/<category>/<preset>/seed_<seed>`, incrementing from the config seed. `load_config()` is declarative and keeps sampler specs intact; `realize_simulation_config()` turns one config plus one seed into the concrete runtime values used for a solve. Periodic constraints are activated with boundary `periodic` operators and resolved against built-in or custom `domain.periodic_maps`. Shared scalar Robin is supported; vector boundaries use shared Dirichlet/Neumann or preset-specific types such as Maxwell `absorbing`, while generic vector Robin remains intentionally unsupported.
- **Domains** (`plm_data/core/mesh.py`) — Domain creation is registry-backed. The current repo ships `interval`, `rectangle`, `box`, `disk`, `dumbbell`, `parallelogram`, `channel_obstacle`, `y_bifurcation`, `venturi_channel`, `porous_channel`, `serpentine_channel`, `l_shape`, `airfoil_channel`, and `annulus`. The Gmsh-backed domains (`disk`, `dumbbell`, `channel_obstacle`, `y_bifurcation`, `venturi_channel`, `porous_channel`, `serpentine_channel`, `l_shape`, `airfoil_channel`, `annulus`) require `python-gmsh`.
- **Output** (`plm_data/core/output.py`) — `FrameWriter` validates requested outputs against the preset spec, expands vector outputs into components for grid formats, and writes post-run stagnation diagnostics to `frames_meta.json`, skipping outputs listed in `static_fields`.

## Usage

Use the `run.sh` wrapper script, which activates the `fenicsx-env` conda environment and launches one single-threaded MPI rank per requested physical core:

```bash
# Run a simulation (single core)
./run.sh run configs/basic/heat/2d_localized_blob_diffusion.yaml --output-dir ./output

# Run a simulation on 4 pinned physical cores
./run.sh -n 4 run configs/basic/heat/2d_localized_blob_diffusion.yaml --output-dir ./output

# Run the same config many times with incrementing seeds from the YAML config
./run.sh -n 4 run configs/basic/heat/2d_localized_blob_diffusion.yaml --output-dir ./output --n-runs 100

# List available presets
./run.sh list

# Build one HTML gallery for all GIF outputs under a directory
./run.sh gallery ./output

# Use a different conda environment if needed
PLM_CONDA_ENV=fenicsx-env-complex ./run.sh run configs/basic/heat/2d_localized_blob_diffusion.yaml --output-dir ./output

# Reserve physical cores 0-3 for one simulation and 4-7 for another
./run.sh -n 4 --core-list 0-3 run configs/basic/heat/2d_localized_blob_diffusion.yaml --output-dir ./output
./run.sh -n 4 --core-list 4-7 run configs/basic/heat/2d_localized_blob_diffusion.yaml --output-dir ./output
```

`-n` now means "use N physical cores" rather than "use N MPI ranks plus extra OpenMP threads". The wrapper forces threaded math backends to one thread and pins MPI ranks to physical cores. When you want multiple simulations to coexist on the same machine without fighting over the same cores, pass `--core-list` with physical core ids from `lscpu -e=CPU,CORE`.

Single runs always clear `<output-dir>/<category>/<preset>` before writing. Batch runs with `--n-runs` clear that preset directory once, then write each run into `seed_<seed>` subdirectories using the config seed and consecutive increments.

The gallery command scans recursively for `.gif` files, groups them by output
directory, and writes `pde_gif_gallery.html` into the scanned directory by
default. Use `--output` to choose a different HTML path.

`maxwell_pulse` runs in the standard real-valued build. The `PLM_CONDA_ENV`
override remains available if you want to activate a different environment
explicitly.

## Adding a new PDE

1. Create a preset class in `plm_data/presets/<category>/` that implements `spec` and `build_problem(config)`.
2. Define a `PresetSpec` with explicit parameters, config-facing coefficients/inputs, boundary-condition fields/operators, solved states, selectable outputs, explicit `static_fields`, and supported dimensions.
3. Return a runtime problem object backed by one of the shared engines, or `CustomProblem` if the PDE needs bespoke solve logic.
4. Register it with `@register_preset("name")`.
5. Create a YAML config in `configs/<category>/<name>/` using the top-level `coefficients:`, `inputs:`, `boundary_conditions:`, and `output.fields:` schema.

## Periodic domains

Periodic constraints are activated on boundary-condition entries:

```yaml
boundary_conditions:
  velocity:
    x-:
      - operator: periodic
        pair_with: x+
    x+:
      - operator: periodic
        pair_with: x-
```

Built-in domains provide the standard face-pair maps. Use `domain.periodic_maps`
only when you need a custom affine slave/master pairing for a domain. Non-periodic
faces still need explicit boundary conditions when the preset requires them.

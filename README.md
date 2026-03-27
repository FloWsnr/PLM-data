# PLM-data

Generate PDE simulation datasets using [DOLFINx](https://github.com/FEniCS/dolfinx) (FEniCSx). Each simulation produces numpy arrays on a regular grid, suitable for training machine learning models.

## Installation

### DOLFINx (FEniCSx)

Create a conda environment with DOLFINx and its dependencies:

```bash
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich  # Linux and macOS
```

See the [DOLFINx README](https://github.com/FEniCS/dolfinx#installation) for alternative installation methods (Docker, apt, Spack, Windows).

### Python dependencies

With the conda environment activated, install the remaining packages:

```bash
pip install pyyaml pytest pyright
# Required for configs that use periodic boundary operators
pip install dolfinx_mpc
# Required for mp4/video output
conda install -c conda-forge ffmpeg
```

## Architecture

- **Presets** (`plm_data/presets/`) — Each PDE is a `PDEPreset` with a `PresetSpec` and a `build_problem(config)` factory. Specs now separate config-facing `inputs`, boundary-condition fields/operators, solved `states`, selectable `outputs`, and explicit `static_fields` that are excluded from post-run stagnation warnings.
- **Problem engines** (`plm_data/presets/base.py`) — Reusable runtime engines cover stationary linear, transient linear, transient nonlinear, and custom problems. Presets still own the formulation details and use shared runtime loops where that helps.
- **Config** (`configs/`) — YAML uses explicit top-level `inputs:`, `boundary_conditions:`, and `output:` sections. Inputs configure sources and initial conditions; `boundary_conditions` configures each preset boundary field; `output.fields` selects which declared outputs to save and how to expand them. Time-dependent presets use a `time:` section; output resolution lives under `output.resolution`. Periodic constraints are activated with boundary `periodic` operators and resolved against built-in or custom `domain.periodic_maps`. Shared scalar Robin is supported; vector boundaries use shared Dirichlet/Neumann or preset-specific types such as Maxwell `absorbing`, while generic vector Robin remains intentionally unsupported.
- **Domains** (`plm_data/core/mesh.py`) — Built-in domains are registry-backed. The current repo ships `interval`, `rectangle`, and `box`.
- **Output** (`plm_data/core/output.py`) — `FrameWriter` validates requested outputs against the preset spec, expands vector outputs into components for grid formats, and writes post-run stagnation diagnostics to `frames_meta.json`, skipping outputs listed in `static_fields`.

## Usage

Use the `run.sh` wrapper script, which activates the `fenicsx-env` conda environment and optionally launches with MPI:

```bash
# Run a simulation (single core)
./run.sh run configs/basic/poisson/2d_default.yaml

# Run a simulation with multiple MPI ranks
./run.sh -n 4 run configs/basic/poisson/2d_default.yaml

# List available presets
./run.sh list

# Use a different conda environment, e.g. a complex-valued DOLFINx build
PLM_CONDA_ENV=fenicsx-env-complex ./run.sh run configs/physics/maxwell/2d_default.yaml
```

The transient `maxwell_pulse` preset runs in the standard real-valued build. The
time-harmonic `maxwell` preset requires a complex-valued DOLFINx/PETSc environment.

## Adding a new PDE

1. Create a preset class in `plm_data/presets/<category>/` that implements `spec` and `build_problem(config)`.
2. Define a `PresetSpec` with explicit parameters, config-facing inputs, boundary-condition fields/operators, solved states, selectable outputs, explicit `static_fields`, and supported dimensions.
3. Return a runtime problem object backed by one of the shared engines, or `CustomProblem` if the PDE needs bespoke solve logic.
4. Register it with `@register_preset("name")`.
5. Create a YAML config in `configs/<category>/<name>/` using the top-level `inputs:`, `boundary_conditions:`, and `output.fields:` schema.

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

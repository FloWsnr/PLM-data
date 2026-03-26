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
```

## Architecture

- **Presets** (`plm_data/presets/`) — Each PDE is a `PDEPreset` with a `PresetSpec` and a `build_problem(config)` factory. Presets are responsible for equations and discretization, not for config validation or output expansion.
- **Problem engines** (`plm_data/presets/base.py`) — Reusable runtime engines cover stationary linear, transient linear, transient nonlinear, and custom problems. Common PDE families share lifecycle code; unusual PDEs use the custom escape hatch.
- **Config** (`configs/`) — YAML is field-centric. Each field owns its `boundary_conditions`, `source`, `initial_condition`, and explicit `output` mode. Time-dependent presets use a `time:` section; output resolution lives under `output.resolution`.
- **Output** (`plm_data/core/output.py`) — `FrameWriter` validates exported base fields against the preset spec, expands vector fields into component arrays, and saves `.npy` outputs.

## Usage

Use the `run.sh` wrapper script, which activates the `fenicsx-env` conda environment and optionally launches with MPI:

```bash
# Run a simulation (single core)
./run.sh run configs/basic/poisson/2d_default.yaml

# Run a simulation with multiple MPI ranks
./run.sh -n 4 run configs/basic/poisson/2d_default.yaml

# List available presets
./run.sh list
```

## Adding a new PDE

1. Create a preset class in `plm_data/presets/<category>/` that implements `spec` and `build_problem(config)`.
2. Define a `PresetSpec` with explicit parameters, fields, supported dimensions, and output modes.
3. Return a runtime problem object backed by one of the shared engines, or `CustomProblem` if the PDE needs bespoke solve logic.
4. Register it with `@register_preset("name")`.
5. Create a YAML config in `configs/<category>/<name>/` using the field-centric schema.

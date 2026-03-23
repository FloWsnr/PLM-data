# PLM-data

Generate PDE simulation datasets using [DOLFINx](https://github.com/FEniCS/dolfinx) (FEniCSx). Each simulation produces numpy arrays on a regular grid, suitable for training machine learning models.

## Architecture

- **Presets** (`plm_data/presets/`) — Each PDE is a Python class that owns its full solve logic (mesh, function space, variational form, solver, time-stepping). A thin base class provides a standard `run(config, output)` interface.
- **Config** (`configs/`) — YAML files specify all parameters explicitly: physical params, mesh/output resolution, domain size, initial conditions, time-stepping, and output settings. No hidden defaults.
- **Output** (`plm_data/core/output.py`) — A `FrameWriter` interpolates FEM solutions onto regular grids and saves them as `.npy` arrays.

## Usage

```bash
# Run a simulation
python -m plm_data run configs/basic/poisson/2d_default.yaml

# List available presets
python -m plm_data list
```

## Adding a new PDE

1. Create a preset class in `plm_data/presets/<category>/` that subclasses `PDEPreset` (or `SteadyLinearPreset` / `TimeDependentPreset` for common patterns)
2. Register it with `@register_preset("name")`
3. Create a YAML config in `configs/<category>/<name>/`

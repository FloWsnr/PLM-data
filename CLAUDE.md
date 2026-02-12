# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDE Simulation Dataset Generator - A modular Python framework for generating 1D/2D/3D PDE simulation trajectories using the `py-pde` library. Generates PNG frame sequences with JSON metadata for training vision-language models.

## Commands

```bash
# Install dependencies
uv sync

# Run all tests (uses 6 parallel workers by default via pytest-xdist)
uv run pytest -q

# Run a specific test file
uv run pytest -q tests/test_simulation.py

# Run single test (disable parallelism for single tests)
uv run pytest -q tests/test_simulation.py::TestSimulationRunner::test_run_short_simulation -n 0

# Run with different worker count
uv run pytest tests/ -v -n auto  # auto-detect CPU cores
uv run pytest tests/ -v -n 0     # disable parallelism

# List all available PDEs
uv run python -m pde_sim list
uv run python -m pde_sim list --category biology

# Show PDE preset info
uv run python -m pde_sim info gray-scott

# Run a simulation
uv run python -m pde_sim run configs/physics/gray_scott/default.yaml
uv run python -m pde_sim run config.yaml --output-dir ./my_output --seed 42

# Run with file-backed frame storage (for large runs)
uv run python -m pde_sim run config.yaml --storage file

# Batch run all configs in a directory
uv run python -m pde_sim batch configs/physics/gray_scott/ --log-file logs/gray_scott.log

# Batch run with parallel execution (6 processes)
uv run python -m pde_sim batch configs/physics/gray_scott/ -np 6

# Generate HTML overview of all GIF simulations
uv run python -m pde_sim overview output/
uv run python -m pde_sim overview output/ --html output/custom_overview.html --title "My Sims"
```

## Architecture

### PDE Registry Pattern
All PDEs use the `@register_pde("name")` decorator for auto-registration. Retrieve with `pde_sim.pdes.get_pde_preset(name)`.

### Base Class Hierarchy
- **PDEPreset** (abstract): Base for all PDEs - defines `metadata`, `create_pde()`, `create_initial_state()`, and `supported_dimensions` validation
- **ScalarPDEPreset**: Single-field systems (heat, wave, etc.)
- **MultiFieldPDEPreset**: Multi-field systems (Gray-Scott u/v, FitzHugh-Nagumo, etc.)

### Simulation Pipeline
1. Load YAML config → `SimulationConfig` dataclass
2. Get PDE preset from registry
3. Create grid with boundary conditions
4. Create PDE object with parameters
5. Generate initial state via IC generator
6. Run time-stepping (py-pde solver)
7. Stream frames to output handlers (png/gif/mp4/numpy/h5) and write metadata (JSON)

### Key Directories
- `pde_sim/core/` - Simulation infrastructure (config, runner, diagnostics, logging, batch)
- `pde_sim/output/` - Output subsystem (handlers/manager/metadata). `pde_sim.core.output` re-exports for compatibility.
- `pde_sim/pdes/` - PDE presets organized by category (basic, biology, physics, fluids)
- `pde_sim/descriptions/` - Markdown description files for each PDE (see below)
- `pde_sim/initial_conditions/` - Initial condition generators
- `pde_sim/boundaries/` - Boundary condition factory

### PDE Descriptions
The `pde_sim/descriptions/` folder contains detailed markdown files for each PDE preset. These files are:
- Named to match the preset name (e.g., `gray-scott.md` for the `gray-scott` preset)
- Automatically loaded and included in the output `metadata.json`
- Contain mathematical formulation, physical background, parameter explanations, and references

When adding a new PDE, create a corresponding `.md` file in `pde_sim/descriptions/` with the same name as the preset.

## Adding a New PDE

1. Create file in appropriate category under `pde_sim/pdes/{category}/`
2. Inherit from `ScalarPDEPreset` or `MultiFieldPDEPreset`
3. Implement `metadata` property and `create_pde()` method
4. Use `@register_pde("name")` decorator
5. Import in `pde_sim/pdes/{category}/__init__.py`
6. Create description file in `pde_sim/descriptions/{name}.md`
7. Add tests in `tests/test_pdes/{category}/test_{name}.py`

Example:
```python
from pde_sim.pdes.base import ScalarPDEPreset, PDEMetadata, PDEParameter
from pde_sim.pdes import register_pde

@register_pde("my-pde")
class MyPDEPreset(ScalarPDEPreset):
    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="My PDE",
            category="physics",
            description="My custom PDE",
            equations={"u": "D * laplace(u)"},
            parameters=[PDEParameter(name="D", description="diffusion coefficient")],
            num_fields=1,
            field_names=["u"],
            supported_dimensions=[1, 2, 3],  # Required: specify which dimensions are supported
        )

    def create_pde(self, parameters: dict, bc, grid):
        # Return py-pde PDE instance
        ...
```

Always add corresponding tests to validate behavior. At least one test should run an actual simulation and check the result.
The result should be finite and not NaN or Inf.


## Configuration Format

YAML configs include: `preset`, `parameters`, `init` (initial conditions), `solver` (euler/rk4), `t_end`, `dt`, `resolution`, `domain_size`, `bc` (boundary conditions), `output` settings.

### Resolution and Domain Size

Resolution and domain_size must be arrays matching the simulation dimensionality:

```yaml
# 1D simulation
resolution: [128]
domain_size: [10.0]

# 2D simulation
resolution: [128, 128]
domain_size: [10.0, 10.0]

# 3D simulation
resolution: [64, 64, 64]
domain_size: [10.0, 10.0, 10.0]
```

### Boundary Conditions

Specify boundaries using py-pde notation. The required boundaries depend on dimensionality:
- **1D**: `x-` (left), `x+` (right)
- **2D**: `x-` (left), `x+` (right), `y-` (bottom), `y+` (top)
- **3D**: `x-`, `x+`, `y-`, `y+`, `z-` (back), `z+` (front)

Simple boundary conditions (same for all fields):
```yaml
bc:
  x-: periodic
  x+: periodic
  y-: neumann:0
  y+: neumann:0
```

Per-field boundary conditions (for multi-field PDEs like thermal-convection):
```yaml
bc:
  x-: periodic
  x+: periodic
  y-: neumann:0
  y+: neumann:0
  fields:               # Per-field overrides (optional)
    omega:
      y-: dirichlet:0
      y+: dirichlet:0
    b:
      y-: neumann:0.2   # Heat flux at bottom
```

Supported BC types:
- `periodic` - Periodic boundary (no value needed)
- `neumann:VALUE` - Fixed derivative (value required, e.g., `neumann:0`)
- `dirichlet:VALUE` - Fixed value (value required, e.g., `dirichlet:0`)

### Output Configuration

Output supports: PNG images, MP4 videos, GIF animations, numpy arrays, and HDF5 trajectories. You can specify one or more formats simultaneously.

```yaml
output:
  path: ./output
  num_frames: 100
  formats: [png]        # Required: list of formats ("png", "mp4", "gif", "numpy", "h5")
  fps: 30               # Frame rate for MP4/GIF (default: 30)
  storage: memory       # "memory" (default) or "file" (file-backed py-pde storage to reduce RAM use)
  keep_storage: false   # Keep intermediate storage file when storage: file
  unique_suffix: false  # Append a short random suffix to run folder names (avoid collisions)
```

**Multiple formats example:**
```yaml
output:
  formats: [png, numpy]  # Generates both PNG frames and numpy arrays
  # or
  formats:
    - png
    - mp4
    - gif
    - numpy
    - h5
```

**Format options:**
- `png`: PNG images per field in subdirectories
- `mp4`: One MP4 video per field
- `gif`: One looping GIF animation per field
- `numpy`: `trajectory.npy` + `times.npy` with shape `(T, *spatial, F)`
- `h5`: `trajectory.h5` with datasets `trajectory` and `times`

All fields from the PDE are always output. Colormaps are auto-assigned from a cycle (viridis, plasma, inferno, magma, cividis) based on field order.

## Output Structure

### PNG Format (default)
```
output/{preset-name}/{config-name}_{run-number}/
├── frames/
│   ├── u/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   └── v/
│       ├── 000000.png
│       ├── 000001.png
│       └── ...
└── metadata.json
```

### MP4 Format
```
output/{preset-name}/{config-name}_{run-number}/
├── u.mp4
├── v.mp4
└── metadata.json
```

### GIF Format
```
output/{preset-name}/{config-name}_{run-number}/
├── u.gif
├── v.gif
└── metadata.json
```

### Numpy Format
```
output/{preset-name}/{config-name}_{run-number}/
├── trajectory.npy    # Shape: (T, H, W, F)
├── times.npy         # Shape: (T,)
└── metadata.json
```

### Multiple Formats
When using `formats: [png, numpy]`:
```
output/{preset-name}/{config-name}_{run-number}/
├── frames/
│   ├── u/
│   │   └── *.png
│   └── v/
│       └── *.png
├── trajectory.npy
├── times.npy
└── metadata.json
```

With explicit `--output-dir`, numbered folders are placed directly in the specified directory:
```bash
uv run python -m pde_sim run configs/biology/immunotherapy/high_effector_diffusion.yaml \
    --output-dir output/biology/immunotherapy
# Creates: output/biology/immunotherapy/high_effector_diffusion_001/
# Next run: output/biology/immunotherapy/high_effector_diffusion_002/
```


### HTML Overview

`pde_sim overview` scans an output directory for simulation folders containing GIF files, copies the GIFs into a self-contained `00_overview/` subdirectory, and generates an HTML page that displays all simulations grouped by preset. Each simulation row shows its name, resolution, time range, frame count, and animated GIF for every field. Requires simulations to have been run with `formats: [gif]` (or including `gif`).

```
output/00_overview/
├── overview.html
├── gray-scott/
│   └── default_001/
│       ├── u.gif
│       └── v.gif
└── ...
```

## New Code Rules

- Don't account for backwards compatibility
- Don't use default values, i.e. dict.get(key, default) or in functions if not actually necessary. The code should fail if values are not provided.
- Never delete outputs (results) without asking
- Never revert git changes you think might be accidental before confirming with the user
- Don't account for backwards compatibility, except when asked. The code should fail if not adhering to the new format.

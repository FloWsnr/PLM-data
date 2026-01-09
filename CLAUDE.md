# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDE Simulation Dataset Generator - A modular Python framework for generating 2D PDE simulation trajectories using the `py-pde` library. Generates PNG frame sequences with JSON metadata for training vision-language models.

## Commands

```bash
# Install dependencies
uv sync

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pdes/test_basic.py -v

# Run single test
pytest tests/test_pdes/test_basic.py::test_heat_metadata -v

# List all available PDEs
python -m pde_sim list
python -m pde_sim list --category biology

# Show PDE preset info
python -m pde_sim info gray-scott

# Run a simulation
python -m pde_sim run configs/defaults/gray_scott_spots.yaml
python -m pde_sim run config.yaml --output-dir ./my_output --seed 42
```

## Architecture

### PDE Registry Pattern
All PDEs use the `@register_pde("name")` decorator for auto-registration. Retrieve with `pde_sim.pdes.get_pde_preset(name)`.

### Base Class Hierarchy
- **PDEPreset** (abstract): Base for all PDEs - defines `metadata`, `create_pde()`, `create_initial_state()`, `get_default_parameters()`, `validate_parameters()`
- **ScalarPDEPreset**: Single-field systems (heat, wave, etc.)
- **MultiFieldPDEPreset**: Multi-field systems (Gray-Scott u/v, FitzHugh-Nagumo, etc.)

### Simulation Pipeline
1. Load YAML config → `SimulationConfig` dataclass
2. Get PDE preset from registry
3. Create grid with boundary conditions
4. Create PDE object with parameters
5. Generate initial state via IC generator
6. Run time-stepping (py-pde solver)
7. Save frames (PNG) and metadata (JSON) via OutputManager

### Key Directories
- `pde_sim/core/` - Simulation infrastructure (config, runner, output)
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
7. Add tests in `tests/test_pdes/test_{category}.py`

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
            equation="∂u/∂t = ...",
            parameters=[PDEParameter("D", "diffusion", 0.01, 1.0, 0.1)],
            field_names=["u"],
        )

    def create_pde(self, parameters: dict, grid):
        # Return py-pde PDE instance
        ...
```

Always add corresponding tests to validate behavior. At least one test should run an actual simulation and check the result.
The result should be finite and not NaN or Inf.


## Configuration Format

YAML configs include: `preset`, `parameters`, `init` (initial conditions), `solver` (euler/rk4), `timesteps`, `dt`, `resolution`, `bc` (boundary conditions), `output` settings.

### Boundary Conditions

Always specify all 4 boundaries using py-pde notation:
- `x-` (left), `x+` (right), `y-` (bottom), `y+` (top)

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

### Output Fields

Specify which fields to output and their colormaps using the `fields` list:

```yaml
output:
  path: ./output
  num_frames: 100
  fields:
    - u:viridis
    - v:plasma
```

Each field is saved with a `{field}_{frame:06d}.png` naming scheme (e.g., `u_000000.png`, `v_000000.png`).

For single-field PDEs or to output only one field:
```yaml
output:
  fields:
    - u:turbo
```

If no `fields` list is specified, all fields are output using the default colormap (`turbo`).

## Output Structure

```
output/{preset-name}/{run-number}/
├── frames/
│   ├── u_000000.png
│   ├── v_000000.png
│   ├── u_000001.png
│   ├── v_000001.png
│   └── ...
└── metadata.json           # Contains per-field colormap info
```

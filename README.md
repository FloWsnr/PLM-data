# PDE Simulation Dataset Generator

A modular Python framework for generating 1D/2D/3D PDE simulation trajectories using [py-pde](https://py-pde.readthedocs.io/), designed for training vision-language models.

## Features

- **14+ PDE presets** across 4 categories (basic, biology, physics, fluids)
- **YAML-based configuration** for easy parameter sweeps
- **Modular architecture** for adding new PDEs
- **PNG frame output** with JSON metadata
- **Slurm integration** for parallel batch processing

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd PLM-data

# Install with uv
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### List Available PDE Presets

```bash
python -m pde_sim list
```

### Get Info About a Preset

```bash
python -m pde_sim info gray-scott
```

### Run a Simulation

```bash
python -m pde_sim run configs/physics/gray_scott/default.yaml
```

### Run with Custom Output Directory

```bash
# Numbered folders are placed directly in the specified directory
python -m pde_sim run configs/biology/immunotherapy/high_effector_diffusion.yaml \
    --output-dir output/biology/immunotherapy
# Creates: output/biology/immunotherapy/high_effector_diffusion_001/
# Next run: output/biology/immunotherapy/high_effector_diffusion_002/
```

## Configuration

Simulations are configured via YAML files:

```yaml
preset: gray-scott          # PDE preset name

parameters:                 # PDE-specific parameters
  F: 0.04                   # Feed rate
  k: 0.06                   # Kill rate
  Du: 0.16                  # Diffusion of u
  Dv: 0.08                  # Diffusion of v

init:                       # Initial condition
  type: gaussian-blob       # IC type
  params:                   # IC parameters
    num_blobs: 10
    width: 0.03

solver: euler               # euler or rk4
t_end: 10000.0              # End time
dt: 1.0                     # Time step size
resolution: [128, 128]      # Grid resolution (must be array)
domain_size: [2.5, 2.5]     # Physical domain size (must be array)

bc:                         # Boundary conditions (use x-, x+, y-, y+)
  x-: periodic
  x+: periodic
  y-: periodic
  y+: periodic

# Per-field boundary conditions (for multi-field PDEs)
# bc:
#   x-: periodic
#   x+: periodic
#   y-: neumann:0
#   y+: neumann:0
#   fields:                 # Per-field overrides
#     omega:
#       y-: dirichlet:0
#       y+: dirichlet:0

output:
  path: ./output            # Output directory
  num_frames: 100           # Total number of frames to save
  fields:                   # Fields to output with colormaps
    - u:viridis             # Field u with viridis colormap
    - v:plasma              # Field v with plasma colormap
# If fields is omitted, all fields are output with default colormap (turbo)

seed: 123                   # Random seed (optional)
```

## Available PDEs

### Basic (4)
- `heat` - Heat diffusion equation
- `inhomogeneous-heat` - Heat with source term
- `wave` - Wave equation
- `advection` - Advection-diffusion equation

### Mathematical Biology (5)
- `schnakenberg` - Turing pattern formation
- `brusselator` - Oscillating reaction-diffusion
- `fisher-kpp` - Population dynamics with traveling waves
- `fitzhugh-nagumo` - Excitable media (spirals, waves)
- `allen-cahn` - Bistable fronts

### Nonlinear Physics (4)
- `gray-scott` - Complex pattern formation
- `cahn-hilliard` - Phase separation
- `swift-hohenberg` - Pattern formation
- `burgers` - Shock wave formation

## Initial Conditions

Available initial condition types:
- `random-uniform` - Uniform random values
- `random-gaussian` - Gaussian random values
- `gaussian-blob` - Gaussian blob(s) with optional aspect ratio for asymmetric shapes
- `sine` - Sinusoidal patterns
- `step` - Step functions

## Output Format

Each simulation creates a directory. Default output (no `--output-dir`):
```
output/{preset-name}/{config-name}_{run-number}/
├── frames/
│   ├── u_000000.png
│   ├── v_000000.png
│   ├── u_000001.png
│   ├── v_000001.png
│   └── ...
└── metadata.json
```

With `--output-dir`, numbered folders go directly in the specified directory:
```
{output-dir}/{config-name}_{run-number}/
```

The `metadata.json` contains:
- Simulation ID and timestamp
- PDE equations and parameters
- Initial conditions and boundary conditions
- Frame annotations with simulation times
- Visualization settings

## Batch Processing with Slurm

For running many simulations in parallel:

```bash
# Generate config files for parameter sweep
# (create configs/batch/config_001.yaml, config_002.yaml, ...)

# Submit array job
sbatch scripts/slurm/array_job.sh configs/batch/
```

Example Slurm script (`scripts/slurm/array_job.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=pde-sim
#SBATCH --array=1-100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00

source .venv/bin/activate
CONFIG_FILE=$(ls configs/batch/*.yaml | sed -n "${SLURM_ARRAY_TASK_ID}p")
python -m pde_sim run ${CONFIG_FILE} --seed ${SLURM_ARRAY_TASK_ID}
```

## Adding New PDEs

1. Create a new file in `pde_sim/pdes/<category>/`:

```python
from pde import PDE, CartesianGrid
from ..base import ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde

@register_pde("my-pde")
class MyPDE(ScalarPDEPreset):
    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="my-pde",
            category="physics",
            description="My custom PDE",
            equations={"u": "D * laplace(u) + f(u)"},
            parameters=[
                PDEParameter("D", 1.0, "Diffusion coefficient"),
            ],
            num_fields=1,
            field_names=["u"],
            supported_dimensions=[1, 2, 3],  # Required: which dimensions this PDE supports
        )

    def create_pde(self, parameters, bc, grid):
        D = parameters.get("D", 1.0)
        return PDE(
            rhs={"u": f"{D} * laplace(u)"},
            bc="periodic",
        )
```

2. Import in the category's `__init__.py`
3. Create a description file in `pde_sim/descriptions/{name}.md`
4. Add tests in `tests/test_pdes/test_{category}.py`
5. Run `python -m pde_sim list` to verify

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pdes/test_basic.py -v
```

## Project Structure

```
pde_sim/
├── __init__.py
├── __main__.py          # CLI entry point
├── core/
│   ├── config.py        # YAML configuration
│   ├── simulation.py    # Simulation runner
│   └── output.py        # PNG/JSON output
├── pdes/
│   ├── base.py          # PDEPreset base classes
│   ├── basic/           # Heat, wave, advection
│   ├── biology/         # Reaction-diffusion, excitable
│   ├── physics/         # Gray-Scott, Cahn-Hilliard
│   └── fluids/          # (placeholder for future)
├── initial_conditions/  # IC generators
└── boundaries/          # BC handling
```

## License

MIT

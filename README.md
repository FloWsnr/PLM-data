# PDE Simulation Dataset Generator

A modular Python framework for generating 1D/2D/3D PDE simulation trajectories using [py-pde](https://py-pde.readthedocs.io/), designed for training vision-language models.

## Features

- **60+ PDE presets** across 4 categories (basic, biology, physics, fluids)
- **YAML-based configuration** for easy parameter sweeps
- **Modular architecture** for adding new PDEs
- **PNG/GIF/MP4/numpy/HDF5 output** with JSON metadata
- **Slurm integration** for parallel batch processing

## Project Layout

- `pde_sim/core/` - config parsing, simulation runner, diagnostics, logging, CLI helpers
- `pde_sim/output/` - output handlers (png/gif/mp4/numpy/h5), output manager, metadata builder
- `pde_sim/pdes/` - PDE presets (basic/biology/physics/fluids)
- `pde_sim/initial_conditions/` - initial condition generators
- `pde_sim/boundaries/` - boundary/grid helpers

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

### Run Large Simulations (Avoid RAM Blowups)

Use file-backed storage for py-pde frames (two-pass streaming output; deletes the intermediate file by default):

```bash
python -m pde_sim run configs/physics/gray_scott/default.yaml --storage file
```

### Run a Batch of Configs

```bash
python -m pde_sim batch configs/physics/gray_scott/ --log-file logs/gray_scott.log

# Run in parallel across 6 processes
python -m pde_sim batch configs/physics/gray_scott/ -np 6
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
  a: 0.037                  # Feed rate
  b: 0.06                   # Kill rate
  D: 2.0                    # Diffusion ratio (v diffuses D times faster than u)

init:                       # Initial condition
  type: gaussian-blob       # IC type
  params:                   # IC parameters
    num_blobs: 10
    width: 0.03

solver: euler               # euler or rk4
t_end: 10000.0              # End time
dt: 1.0                     # Time step size
resolution: [128, 128]      # Grid resolution (must be array)
domain_size: [1000, 1000]   # Physical domain size (must be array)

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
  formats: [png]            # List of formats: "png", "gif", "mp4", "numpy", "h5" (can combine)
  fps: 30                   # Frame rate for MP4 (default: 30)
  storage: memory           # "memory" (default) or "file" to avoid keeping all frames in RAM
  keep_storage: false       # Keep intermediate storage file when storage: file
  unique_suffix: false      # Append a short random suffix to run folder names (avoid collisions)
# All fields are automatically output with auto-assigned colormaps
# Multiple formats example: formats: [png, numpy]

seed: 123                   # Random seed (optional)
```

## Available PDEs

Run `python -m pde_sim list` to see all 60+ presets. Examples by category:

### Basic (12)
- `heat`, `wave`, `advection`, `plate`, `schrodinger`
- `inhomogeneous-heat`, `inhomogeneous-wave`, `damped-wave`

### Mathematical Biology (20)
- `brusselator`, `schnakenberg`, `gierer-meinhardt`, `lotka-volterra`
- `fisher-kpp`, `fitzhugh-nagumo`, `allen-cahn`, `keller-segel`
- `klausmeier`, `sir`, `immunotherapy`, `cyclic-competition`

### Nonlinear Physics (20)
- `gray-scott`, `cahn-hilliard`, `swift-hohenberg`, `kuramoto-sivashinsky`
- `burgers`, `kdv`, `sine-gordon`, `complex-ginzburg-landau`
- `lorenz`, `duffing`, `van-der-pol`, `fokker-planck`

### Fluids (8)
- `navier-stokes`, `vorticity`, `shallow-water`, `thermal-convection`

## Initial Conditions

Available initial condition types:
- `random-uniform` - Uniform random values
- `random-gaussian` - Gaussian random values
- `gaussian-blob` - Gaussian blob(s) with optional aspect ratio for asymmetric shapes
- `sine` / `cosine` - Sinusoidal patterns
- `step` / `double-step` - Step functions and band/stripe patterns
- `rectangle-grid` - Piecewise-constant grid patterns
- `constant` - Uniform field (useful with Dirichlet BCs)

## Output Format

Each simulation creates a directory. Default output (no `--output-dir`):
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
├── trajectory.npy          # if formats includes "numpy"
├── times.npy               # if formats includes "numpy"
├── trajectory.h5           # if formats includes "h5" (datasets: "trajectory", "times")
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
                PDEParameter(name="D", description="Diffusion coefficient"),
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
4. Add tests in `tests/test_pdes/{category}/test_{name}.py`
5. Run `python -m pde_sim list` to verify

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pdes/basic/test_heat.py -v
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
│   └── fluids/          # Navier-Stokes, vorticity, shallow water
├── initial_conditions/  # IC generators
└── boundaries/          # BC handling
```

## License

MIT

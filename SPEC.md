# PLM-data Refactor Specification

## Mission

PLM-data generates diverse two-dimensional, time-dependent PDE simulation data.

The project optimizes for breadth of dynamics, robust random sampling, and simple
operation. It is not trying to be a high-accuracy scientific solver framework or a
general-purpose PDE library. The core workflow is:

1. Randomly sample a compatible simulation setup.
2. Run one DOLFINx simulation.
3. Save grid, mesh, media, animation, and diagnostic outputs.

Curated YAML configs are removed from the architecture. They are not retained as
an import path, developer shortcut, compatibility layer, or reproducibility
format. The runner owns random simulation generation directly.

## Scope

The refactored project supports only 2D, time-dependent PDE simulations.

Remove support for:

- 1D domains and PDEs.
- 3D domains and PDEs.
- PDEs that are not time-dependent.
- YAML config loading, shared YAML fragments, and backward compatibility with
  the current config schema.

Keep support for:

- DOLFINx as the finite-element simulation framework.
- Gmsh as the mesh-generation backend.
- MPI execution.
- Pytest tests.
- Ruff linting and formatting.
- Numpy, VTK, GIF, and video outputs. Video remains supported, but it is not a
  default output format for the first random runner.

## User Interface

`run.sh` is the only user-facing entrypoint. The refactor removes the current
subcommand interface, including YAML `run`, `list`, and `gallery` commands.

The default behavior is a single fully random simulation run. A seed is required
for every run; bare `./run.sh` without `--seed` must fail with a clear error.
No CLI filters for PDE, domain, boundary scenario, initial-condition scenario,
or parameter ranges are required in the first refactor.

The wrapper keeps MPI/core options and adds the required simulation seed:

```bash
./run.sh --seed 1234
./run.sh -n 4 --seed 1234
./run.sh -n 4 --core-list 0-3 --seed 1234
```

The runner uses `./output` as the default output root. An output-root override
may be added if useful, but the first refactor must not require it.

Each run output directory should include sampled simulation information in the
path, for example:

```text
output/<pde>/<domain>/seed_<seed>_<short-run-id>/
```

The exact path format can change, but it must be human-scannable and include at
least the sampled PDE, sampled domain, and seed or run id.

## Core Concepts

The architecture is built around first-class objects:

- PDEs
- domains
- boundary-condition operators
- boundary-condition scenarios
- initial-condition operators
- initial-condition scenarios
- output writers
- samplers and constraints

The current `presets` concept should be renamed to `pdes`. The current
`families` concept should be renamed to `scenarios`.

Operators implement local mathematical behavior. Scenarios choose and arrange
operators into complete simulation pieces.

Examples:

- A boundary operator is `dirichlet`, `neumann`, `robin`, `periodic`, or
  `absorbing`.
- A boundary scenario is `all_neumann`, `periodic_x_walls_y_neumann`, or
  `inlet_drive_outlet_open`.
- An initial-condition operator is `gaussian_bump`, `gaussian_noise`,
  `sine_waves`, or `constant`.
- An initial-condition scenario is `two_blobs`, `noise_plus_modes`, or
  `boundary_layer_blob`.

Scenarios generate concrete runtime configuration. Operators perform assembly,
interpolation, or initialization work.

## Package Layout

Use package names that match the architecture:

```text
plm_data/
  pdes/
    base.py
    registry.py
    heat/
      pde.py
      spec.py
      helpers.py
  domains/
    base.py
    registry.py
    rectangle/
      domain.py
      spec.py
      helpers.py
  boundary_conditions/
    operators/
      dirichlet/
        operator.py
        spec.py
      neumann/
        operator.py
        spec.py
    scenarios/
      all_neumann/
        scenario.py
        spec.py
  initial_conditions/
    operators/
      gaussian_bump/
        operator.py
        spec.py
    scenarios/
      noise_plus_blobs/
        scenario.py
        spec.py
  sampling/
    constraints.py
    context.py
    samplers.py
  core/
    runner.py
    output.py
    health.py
```

The exact file names may vary, but each first-class object must keep its runtime
implementation close to its spec.

## Specs

Every first-class object has a spec. Specs are Python objects, not YAML.

Specs must describe:

- Identity: name, description, category where useful.
- Compatibility: dimensions, field shapes, boundary roles, required operators,
  supported scenarios, solver strategies, and output capabilities.
- Parameters: names, types, constraints, samplers, and dependencies.
- Runtime needs: mesh requirements, time stepping requirements, solver profiles,
  MPI support, and output fields.

Specs should be executable enough that central hard-coded validation blocks are
not needed. Adding a new domain, PDE, operator, or scenario should usually mean
adding a package with a spec and implementation, not editing a long `if type ==`
chain elsewhere.

## Constraint Model

Constraints and samplers are Python functions.

Each sampler receives a shared sampling context containing already chosen
objects and already sampled values. This allows dependent constraints such as:

- Obstacle radius depends on channel height.
- Gaussian centers depend on the sampled domain geometry.
- Mesh size depends on domain scale and PDE stability needs.
- Time step depends on the sampled PDE parameters and mesh scale.
- Boundary values depend on the sampled PDE field shape.

Independent min/max ranges are allowed only when they are genuinely sufficient.
The common case should be contextual sampling.

Sampling should avoid invalid combinations by construction. Validation still
exists, but it is an assertion layer rather than the main rejection mechanism.

The first implementation uses simple uniform random choices among compatible
objects. More complex policies, including weighted sampling, coverage-driven
sampling, curriculum sampling, or user-provided policies, are later extensions;
the interfaces should leave room for them.

## Random Simulation Pipeline

The runner performs one requested random simulation by trying deterministic
sampling attempts derived from the required seed. Invalid sampled combinations
are discarded and resampled up to a fixed attempt budget. The first
implementation can use an internal constant for this budget; it does not need a
CLI surface.

Only a successful attempt is committed to the dataset output tree. Failed
attempt directories and logs are not kept by default. If the attempt budget is
exhausted, the process exits nonzero with a concise failure summary.

For one attempt, the runner:

1. Build registries for all available 2D time-dependent PDEs, domains,
   boundary scenarios, initial-condition scenarios, and operators.
2. Sample a compatible domain/PDE pair.
3. Sample boundary-condition scenarios from the intersection of:
   - domain boundary names and roles,
   - domain-supported boundary scenarios,
   - PDE boundary fields,
   - PDE-supported boundary operators,
   - PDE field shapes,
   - periodic map availability.
4. Sample initial-condition scenarios from the intersection of:
   - domain dimension and coordinate regions,
   - PDE input/state field shapes,
   - PDE-supported initial-condition operators,
   - scenario seed and parameter requirements.
5. Build one combined sampling context.
6. Sample all domain, PDE, boundary, initial-condition, solver, time, output,
   and stochastic parameters from that context.
7. Validate the complete concrete runtime config.
8. Build the Gmsh domain.
9. Build the DOLFINx PDE runtime problem.
10. Run the simulation.
11. Write outputs and diagnostics.

Under MPI, sampling must be rank-consistent. The preferred model is that rank 0
samples and validates one concrete runtime config, then broadcasts it to the
other ranks before mesh generation and solve setup.

## PDEs

A PDE package defines:

- A PDE spec.
- A runtime class that builds spaces, forms, solvers, state, sources, boundary
  terms, time stepping, and output fields.
- Optional helper functions for shared formulations.

PDE specs declare:

- Name and category.
- Equations or short mathematical description.
- Time dependency.
- Supported dimension, always 2D in the refactored system.
- State fields and field shapes.
- Coefficients and parameter specs.
- Input/source fields and supported initial-condition shapes.
- Boundary fields and supported boundary operators.
- Output fields and output modes.
- Static fields excluded from stagnation warnings.
- Solver strategies and solver profiles.
- Mesh and time-step requirements or samplers.
- Stochastic support, if any.

Only 2D time-dependent PDEs remain in the registry.

The first implementation should migrate a small representative set before
expanding to all current 2D PDEs. The representative set must include at least:

- One scalar PDE.
- One vector-field PDE.
- One nonlinear PDE.
- One fluid PDE.

After that vertical slice works end to end, migrate all remaining current 2D
time-dependent PDEs.

## Domains

A domain package defines:

- A domain spec.
- A Gmsh builder.
- Optional geometry helpers and coordinate-region samplers.

All domains are Gmsh-backed.

Domain specs declare:

- Name and description.
- Dimension, always 2D.
- Parameters and contextual samplers.
- Boundary names.
- Boundary roles.
- Periodic boundary pairs and geometric maps, if supported.
- Supported boundary scenarios.
- Supported initial-condition scenarios or coordinate regions.
- Geometry validity constraints.

Domains should expose coordinate-region samplers where useful. Examples:

- interior
- near boundary
- upstream
- downstream
- wake
- near obstacle
- near inner rim
- near outer wall

Initial-condition scenarios should use these domain samplers rather than
hard-coded coordinate ranges.

## Boundary Conditions

Boundary-condition operators define assembly behavior and value requirements.

Boundary-condition operator specs declare:

- Operator name.
- Supported field shapes.
- Value shape, if a value is required.
- Operator parameters.
- Whether a paired boundary is required.
- MPI and periodic-constraint requirements.

Boundary-condition scenarios generate complete boundary configurations after the
domain and PDE are known.

Boundary scenarios use a two-level design:

- Field-level scenarios configure one PDE boundary field, for example one
  scenario for scalar field `u`.
- PDE-level scenarios configure all boundary fields for one PDE/domain pair,
  composing field-level scenarios where possible and overriding them where
  fields must be coordinated.

The random runner samples PDE-level boundary scenarios. Scalar PDEs can expose
simple PDE-level scenarios that wrap one field-level scenario. Coupled PDEs must
use PDE-level scenarios to avoid invalid or unphysical independent choices, such
as incompatible velocity and pressure conditions in incompressible flow, coupled
height and momentum conditions in shallow-water models, or inconsistent wall,
periodic, or absorbing choices for electromagnetic fields.

Boundary-condition scenario specs declare:

- Supported dimensions.
- Required domain roles or boundary names.
- Required periodic pairs, if any.
- Supported PDE field shapes.
- Operators used by the scenario.
- Parameter samplers for boundary values and operator parameters.
- Whether the scenario is field-level or PDE-level.
- For PDE-level scenarios, the PDE boundary fields configured by the scenario
  and any coupling constraints among them.

Scenario sampling must happen after the domain and PDE are known.

## Initial Conditions

Initial-condition operators define field construction behavior.

Initial-condition operator specs declare:

- Operator name.
- Supported dimensions.
- Supported field shapes.
- Parameters and samplers.
- Whether a seed is required.
- Whether vector field-level application is supported.

Initial-condition scenarios combine one or more operators into a concrete field
initialization.

Initial-condition scenarios declare:

- Supported dimensions.
- Supported field shapes.
- Required domain coordinate regions.
- Operators used by the scenario.
- Parameter samplers.

Scenarios should sample valid domain locations through domain coordinate-region
samplers instead of relying on rectangular bounding boxes.

## Solver And Time Sampling

Solver and time settings are part of random generation.

PDE specs should expose allowed solver strategies and recommended serial/MPI
PETSc option profiles. The runner chooses a valid profile based on communicator
size.

Time settings must be sampled or derived from the PDE/domain context:

- `dt`
- `t_end`
- number of output frames
- any PDE-specific stability or health limits

Time-step sampling should prefer stable, diverse simulations over aggressive
parameter ranges that frequently fail.

## Output

Supported output formats:

- Numpy arrays interpolated to regular grids.
- Native VTK output for mesh-based inspection.
- GIF animations generated from grid data.
- Video generated from grid data.

Default random-run formats are:

- `numpy`
- `gif`
- `vtk`

Video output is supported by the writer layer, but it is optional and not part
of the first random-run default.

The output system must:

- Validate requested output fields against the PDE spec.
- Expand vector outputs into component arrays for grid formats.
- Save domain masks for non-rectangular domains where grid points fall outside
  the mesh.
- Check finite values during simulation.
- Write post-run diagnostics, including stagnation checks.
- Skip declared static fields in stagnation warnings.

The output resolution is sampled as part of the run context. The first
implementation can use one conservative 2D resolution range.

## Health And Diagnostics

Runtime health checks are required.

The runner should detect:

- Non-finite values in FEM state vectors.
- Non-finite values in sampled grid outputs.
- Solver non-convergence.
- Excessive or suspicious stagnation.
- Failed mesh generation.
- Failed interpolation.

Each successful run should write metadata containing:

- Status.
- Stage reached.
- Sampled PDE and domain.
- Seed and run id.
- Output directory.
- Solver summary.
- Time-step summary.
- Output summary.
- Health diagnostics.

In a later step, each run should also write the concrete sampled runtime config
or manifest needed to reproduce the run. This reproducibility manifest is not
required in the first random-run implementation.

## Testing

Tests should cover the architecture rather than old YAML compatibility.

Required test groups:

- Registry discovery for PDEs, domains, operators, and scenarios.
- Spec validation.
- Compatibility intersection logic.
- Contextual parameter sampling.
- Concrete runtime-config validation.
- Gmsh mesh generation for each domain.
- Boundary scenario generation for representative PDE/domain pairs.
- Initial-condition scenario generation for representative domains.
- Smoke runs for a small set of sampled simulations.
- MPI smoke coverage for at least one scalar PDE and one vector PDE.
- Output writer tests for numpy, VTK, GIF, video, domain masks, and diagnostics.
- CLI tests that `./run.sh` requires `--seed` and does not expose the removed
  subcommands.
- Retry tests that invalid sampled combinations are discarded and only
  successful attempts are committed.

## Naming

Use the new vocabulary consistently:

- `pde`, not `preset`.
- `scenario`, not `family`.
- `operator` for low-level BC and IC implementations.
- `spec` for compatibility and sampling metadata.
- `runtime config` for the concrete sampled configuration used by a simulation.

## Migration Plan

1. Perform a big-bang rename and deletion pass:
   - Create the new package skeleton: `pdes`, `sampling`, BC scenarios, and IC
     scenarios.
   - Move and rename current `presets` to `pdes`, keeping only 2D
     time-dependent PDEs.
   - Rename boundary-condition and initial-condition `families` to `scenarios`.
   - Delete 1D and 3D domains, PDE support, YAML configs, tests, and
     compatibility code.
   - Delete YAML config loading and shared-fragment support.
2. Move validation and realization logic out of central `if type == ...` blocks
   and into object specs plus shared generic validators.
3. Implement in-memory random runtime-config generation with a required seed,
   rank-consistent MPI sampling, uniform random compatible choices, and bounded
   retry on invalid samples.
4. Implement the no-subcommand `run.sh --seed ...` user interface.
5. Update output directory naming to include sampled PDE/domain/run identity.
6. Keep current output writers and health diagnostics, adapting them to the new
   runtime config. Default random-run outputs are numpy, GIF, and VTK.
7. Migrate a representative PDE set first: scalar, vector-field, nonlinear, and
   fluid. Use this set to validate the architecture end to end.
8. Rewrite tests around random generation, compatibility, specs, retries,
   output writers, and smoke runs.
9. Migrate all remaining current 2D time-dependent PDEs.
10. Add run-manifest writing in a second step so sampled simulations can be
    reproduced exactly.
11. Add optional sampling policies or CLI filters later, after the simple random
    runner is stable.

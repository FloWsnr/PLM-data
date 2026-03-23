---
name: DOLFINx Reference Lookup
description: Look up DOLFINx documentation, demos, examples, and API source code. Use when implementing or debugging DOLFINx-based presets, or when the user asks about DOLFINx APIs and patterns. Invoked via /dolfinx-reference.
---

## DOLFINx Reference Lookup

When you need to understand DOLFINx APIs, patterns, or implementation approaches, use the resources below. This is especially useful when implementing new PDE presets or debugging FEniCSx issues.

### Local Reference (v0.10.0)

The full DOLFINx source is available locally at `reference/dolfinx/`. Always check local files first before fetching web docs.

#### Demos (`reference/dolfinx/python/demo/`)

These are complete, self-contained examples. Each file is a jupytext-format Python script with embedded markdown documentation explaining the math, variational formulation, and implementation.

**Most relevant demos for this project:**

| Demo file | What it demonstrates |
|---|---|
| `demo_poisson.py` | Basic linear PDE, boundary conditions, `LinearProblem` |
| `demo_cahn-hilliard.py` | Nonlinear time-dependent 4th-order PDE, mixed FE, `NonlinearProblem`, theta-method |
| `demo_navier-stokes.py` | Time-dependent Navier-Stokes, DG methods, upwinding |
| `demo_stokes.py` | Mixed elements (Taylor-Hood), block solvers |
| `demo_mixed-poisson.py` | Mixed formulation, block preconditioners |
| `demo_helmholtz.py` | Complex-valued PDE, frequency-domain |
| `demo_biharmonic.py` | 4th-order PDE via interior penalty DG |
| `demo_elasticity.py` | Vector-valued PDE, algebraic multigrid |
| `demo_hdg.py` | Hybridizable DG scheme |
| `demo_poisson_matrix_free.py` | Matrix-free CG solver |
| `demo_interpolation-io.py` | Interpolation between spaces, I/O |
| `demo_gmsh.py` | Custom mesh generation with Gmsh |
| `demo_pyvista.py` | Visualization |
| `demo_scattering_boundary_conditions.py` | EM scattering, absorbing BCs |
| `demo_pml.py` | Perfectly matched layers |
| `demo_half_loaded_waveguide.py` | Electromagnetic modal analysis |
| `demo_axis.py` | Axisymmetric problems |
| `demo_static-condensation.py` | Static condensation of linear elasticity |
| `demo_lagrange_variants.py` | Different Lagrange element variants |
| `demo_tnt-elements.py` | Custom Basix elements |
| `demo_types.py` | PDEs with different scalar types (float32, complex) |
| `demo_pyamg.py` | Using pyamg as a solver |
| `demo_mixed-topology.py` | Mixed-topology meshes |
| `demo_comm-pattern.py` | Parallel communication patterns |

**How to use demos:** Read the demo file directly. The comments contain the full mathematical formulation and step-by-step explanation. Focus on:
- How function spaces are created
- How variational forms (a, L or F) are defined
- How boundary conditions are applied
- How the solver is configured and called
- How time-stepping is structured (for time-dependent problems)

#### Python Source (`reference/dolfinx/python/dolfinx/`)

The actual DOLFINx Python API source code. Key modules:

| Module | Purpose |
|---|---|
| `fem/function.py` | `Function`, `FunctionSpace`, expression evaluation |
| `fem/bcs.py` | `dirichletbc` and boundary condition utilities |
| `fem/forms.py` | Form compilation and assembly |
| `fem/petsc.py` | `LinearProblem`, `NonlinearProblem`, PETSc wrappers |
| `fem/assemble.py` | Low-level assembly routines |
| `fem/element.py` | Finite element definitions |
| `mesh.py` | Mesh creation, topology, geometry |
| `geometry.py` | Bounding box trees, point queries |
| `plot.py` | Plotting utilities |
| `common.py` | Timing, MPI utilities |

#### Tests (`reference/dolfinx/python/test/unit/`)

Unit tests organized by module (fem, mesh, io, geometry, la, nls, etc.). Useful for seeing edge cases and API usage patterns. Key test directories:
- `test/unit/fem/` — assembly, BCs, interpolation, solvers, function spaces
- `test/unit/mesh/` — mesh creation, refinement, partitioning
- `test/unit/nls/` — nonlinear solver tests

### Web Documentation

The online docs at `https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/` provide:
- **API reference** with full docstrings: `https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/api.html`
- **Demo pages** with rendered math and explanations: `https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/demos.html`
- Individual demo: `https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/demos/demo_<name>.html`

Use `WebFetch` to retrieve specific API docs or demo pages when the local source isn't sufficient (e.g., for rendered docstrings or cross-references).

### Lookup Strategy

When implementing or debugging a DOLFINx feature:

1. **Start with demos** — Find the demo closest to what you're implementing and read it. The demos are the best resource for idiomatic DOLFINx patterns.
2. **Check the source** — If you need to understand a specific API (arguments, return types, behavior), read the source in `reference/dolfinx/python/dolfinx/`.
3. **Check tests** — For edge cases or less-documented features, the unit tests in `reference/dolfinx/python/test/unit/` show how the API is exercised.
4. **Fetch web docs** — For rendered API docstrings or when you need cross-referenced documentation, fetch from `https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/`.

### Common Patterns to Look Up

- **Boundary conditions**: `demo_poisson.py` for Dirichlet BCs, `demo_mixed-poisson.py` for natural BCs
- **Time-stepping**: `demo_cahn-hilliard.py` (theta-method), `demo_navier-stokes.py` (IPCS-like)
- **Nonlinear problems**: `demo_cahn-hilliard.py` (Newton solver via `NonlinearProblem`)
- **Mixed function spaces**: `demo_stokes.py` (Taylor-Hood), `demo_cahn-hilliard.py` (mixed element)
- **DG methods**: `demo_biharmonic.py` (interior penalty), `demo_navier-stokes.py` (divergence-conforming DG)
- **Interpolation**: `demo_interpolation-io.py`
- **Custom meshes**: `demo_gmsh.py`

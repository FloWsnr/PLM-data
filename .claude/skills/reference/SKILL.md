---
name: VisualPDE Reference Lookup
description: Look up reference implementations, parameters, and descriptions from the original VisualPDE simulation package. Use this whenever working on a PDE preset, tuning parameters, or changing configs.
---

## Reference Directory

The original VisualPDE reference material lives at:

```
PLM-data/reference/visual-pde/
```

## Step 1: Find the PDE Description

Look up the PDE's markdown description in the appropriate category directory:

| Category | Directory |
|---|---|
| Basic PDEs | `reference/visual-pde/_basic-pdes/` |
| Nonlinear Physics | `reference/visual-pde/_nonlinear-physics/` |
| Mathematical Biology | `reference/visual-pde/_mathematical-biology/` |
| Fluids | `reference/visual-pde/_fluids/` |
| Art PDEs | `reference/visual-pde/_art-pdes/` |

Each `.md` file has Jekyll frontmatter with the PDE name, equations (LaTeX), and a description, followed by rich content explaining the physics/biology.

**Search strategy:** If you don't know which category the PDE belongs to, search across all directories:
```
Glob: reference/visual-pde/_*/*.md
```
Then grep for the PDE name in the results.

## Step 2: Look Up the Preset Configuration in presets.js

The file containing all solver configs, parameter values, and initial conditions:

```
PLM-data/reference/visual-pde/sim/scripts/RD/presets.js
```

This is a large file (~7900 lines, 229 presets). Each preset is a JS object keyed by name:

```javascript
presets["Gray-Scott"] = {
  dt: 0.5,
  domainScale: "1.5",
  spatialStep: "0.005",
  kineticParams: "F = 0.04; k = 0.06; Du = 0.16; Dv = 0.08;",
  reactionStr_1: "-u*v^2 + F*(1-u)",
  reactionStr_2: "u*v^2 - (F+k)*v",
  diffusionStr_1_1: "Du",
  diffusionStr_2_2: "Dv",
  initCond_1: "1 - exp(...)",
  initCond_2: "exp(...)",
  boundaryConditions_1: "periodic",
  ...
};
```

**Search strategy:** Use Grep to find the preset by name:
```
Grep pattern: presets\[".*KEYWORD" in presets.js
```

Then read the surrounding ~50 lines to get the full preset definition.

**Key fields to extract:**
- `kineticParams` — parameter names and reference values
- `reactionStr_*` — reaction/kinetic terms per species
- `diffusionStr_*_*` — diffusion coefficients
- `dt`, `domainScale`, `spatialStep` — solver/grid configuration
- `initCond_*` — initial condition expressions
- `boundaryConditions_*` — boundary condition types
- `numSpecies`, `speciesNames` — field definitions

## Step 3: Compare with Our Implementation

After gathering the reference info, compare against our PDE preset and config:

1. **Equations** — Do our reaction terms match the reference?
2. **Parameters** — Are default values in a reasonable range compared to the reference?
3. **Solver settings** — Is our `dt` appropriate? Check `domainScale` and `spatialStep` for domain/resolution guidance.
4. **Boundary conditions** — Do they match (periodic, neumann, dirichlet)?
5. **Initial conditions** — Are ours physically reasonable compared to the reference?

## Important Notes

- A single PDE may have MULTIPLE presets in `presets.js` (e.g., "Gray-Scott", "Gray-Scott (worms)", "Gray-Scott (U-skate)"). Check all variants.
- The reference uses WebGL shaders for simulation — equations may be written differently but should be mathematically equivalent.
- Parameter names in the reference may differ from ours — focus on matching the mathematical meaning.
- Always report what you found to the user so they can make informed decisions about parameter tuning.

---
name: pde-simulation-analysis
description: Analyze a PLM-data PDE simulation run for numerical health and physical plausibility. Use when a user asks whether a simulation output looks valid, wants a diagnosis of a run directory under output/, or wants post-run analysis. Always inspect frames_meta.json, run_meta.json, and simulation.log, compute PDE-appropriate validity metrics from the saved arrays, and extract and view representative GIF frames. Invoked via /pde-simulation-analysis.
---

## PDE Simulation Analysis

Use this skill when evaluating whether a PLM-data simulation run is numerically healthy and physically plausible.

The typical input is a run directory like `output/<category>/<preset>/`.

### Required Workflow

1. Always open `frames_meta.json` and `run_meta.json` first.
   - Do not skip them, even if the GIFs already look reasonable.
   - From `run_meta.json`, record the run `status`, `stage`, preset/category, solver convergence, frame/timestep counts, health summary, resolved config, and any error payload.
   - From `frames_meta.json`, record the saved `times`, `field_names`, output resolution, expected outputs, and diagnostic results.
   - Treat built-in health checks as a starting point, not the full analysis.

2. Read `simulation.log`.
   - Search for warnings, errors, solver non-convergence, timestep cuts, clipping/flooring, NaN/Inf messages, repeated retries, and suspicious runtime behavior.
   - Correlate log messages with frame times and metadata health flags.
   - If `run_meta.json` says the run failed, use the log to localize the failing stage before looking at field arrays.

3. Compute additional validity metrics from the saved arrays.
   - Do not rely only on the built-in checks in `frames_meta.json`.
   - Always compute basic per-field summaries across time: min, max, mean, standard deviation, spatial integral or sum, L2 norm, finite-value fraction, and frame-to-frame deltas.
   - If `domain_mask.npy` exists, exclude out-of-domain points from all metrics.
   - Then choose PDE-specific metrics that match the preset and boundary conditions.

#### PDE-Specific Metric Ideas

- Conservative transport / compressible Euler / wave-like conservative systems:
  - Mass conservation
  - Momentum or energy trends when relevant
  - Positivity of density and pressure
  - Shock localization without checkerboard artifacts or obvious Gibbs ringing

- Heat / diffusion:
  - Mass conservation when boundary conditions imply zero net flux
  - Decay of variance or L2 energy
  - Monotone smoothing
  - No creation of new extrema for pure diffusion problems

- Reaction-diffusion:
  - Positivity or known bounds
  - Growth or decay rates consistent with the reaction terms
  - Pattern wavelength and amplitude evolution
  - Note explicitly when mass is not expected to be conserved

- Incompressible flow:
  - Divergence-related quantities if recoverable from outputs
  - Kinetic-energy trends consistent with viscosity and forcing
  - Boundary-condition consistency near walls, inlets, and outlets

- Maxwell / Helmholtz / wave problems:
  - Energy or norm balance
  - Damping or resonance expectations
  - Symmetry and boundary-condition consistency

If exact conservation is not expected, say why and pick a better metric. The point is to test the physics the preset is supposed to represent, not to force a generic invariant.

4. Extract representative image frames and inspect them visually.
   - Use the saved GIFs when present.
   - Extract at least the start, middle, and end frames for one or more key fields.
   - Use `scripts/extract_gif_frames.py` in this skill folder for deterministic extraction.
   - Actually open the extracted PNGs with the available image-viewing tool. Do not stop after writing them to disk.
   - Look for boundary leaks, symmetry breaking, checkerboarding, ringing, clipping, dead or static fields, sign mistakes, scale mistakes, and domain-mask artifacts.

5. Synthesize the evidence.
   - Combine metadata, logs, computed metrics, and visual inspection into a final verdict.
   - Separate hard failures from softer concerns.
   - State whether the run looks physically plausible, numerically suspicious, or clearly broken.
   - End with the single most informative next check if uncertainty remains.

### Analyzing multiple simulations

If you analyze multiple runs to compare them, generate a plot which compare the start, middle, and end frames for one (important) field across the runs.
This can be done by extracting the frames as PNGs and then using a plotting library to create a side-by-side comparison.
Look for differences in the patterns, magnitudes, and any artifacts that may indicate differences in the underlying physics or numerical stability of the runs.


### Practical Notes

- Prefer inspecting the most physically interpretable outputs first, such as `density`, `pressure`, `temperature`, `concentration`, `total_energy`, and velocity components or magnitude.
- If only `.npy` outputs exist, render representative PNGs from those arrays and inspect them instead of GIFs.
- If the run failed early, say which parts of the workflow were unavailable rather than pretending the analysis is complete.

### Reporting Format

Report these sections in order:

1. Run summary
2. Metadata findings from `run_meta.json` and `frames_meta.json`
3. Log findings
4. Computed metrics and whether they match the PDE expectations
5. Visual findings from start, middle, and end frames
6. Final verdict and next action

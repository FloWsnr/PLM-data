# Basic PDE Simulation Analysis - FINAL

## Summary

All 7 basic PDEs now work with good visible dynamics over 100 frames.

| PDE | Status | Time | Dynamics | Fixes Applied |
|-----|--------|------|----------|---------------|
| heat | WORKS | 22s | GOOD | D: 0.1 -> 0.05 |
| wave | WORKS | 55s | GOOD | (none needed) |
| advection | WORKS | 38s | GOOD | D: 0.01 -> 0.005 |
| inhomogeneous-heat | WORKS | 13s | EXCELLENT | Converted to real visualpde.com equation with spatial source |
| schrodinger | WORKS | 75s | EXCELLENT | (none needed) |
| plate | WORKS | 75s | EXCELLENT | (none needed) |
| inhomogeneous-wave | WORKS | 23s | EXCELLENT | Converted to real visualpde.com equation with varying wave speed |

## Detailed Analysis

### heat
- Shows 3 gaussian blobs diffusing and merging
- D=0.05 keeps blobs visible through frame 50, faded structure at frame 99
- Good demonstration of heat diffusion

### wave
- Single blob creates expanding circular waves
- Beautiful interference patterns throughout
- No changes needed - well tuned

### advection
- Two blobs translate horizontally while diffusing
- D=0.005 preserves blob visibility longer
- Shows advection-diffusion dynamics clearly

### inhomogeneous-heat
- **CONVERTED** to real visualpde.com equation: dT/dt = D∇²T + f(x,y)
- Source term f(x,y) = D·π²·(n²+m²)/L² · cos(nπx/L)·cos(mπy/L)
- Creates beautiful 2x2 grid pattern as system evolves to steady state
- Initial Gaussian blobs diffuse while cosine source pattern becomes dominant
- Uses Neumann BCs (required for bounded solutions)

### schrodinger
- Wave packet with momentum creates interference patterns
- Most visually complex and interesting
- Beautiful quantum dynamics

### plate
- Constant initial condition with Dirichlet BCs
- Creates beautiful grid-like standing wave patterns
- Excellent demonstration of plate vibration modes

### inhomogeneous-wave
- **CONVERTED** to real visualpde.com equation: d²u/dt² = ∇·(f(x,y)∇u)
- Spatially varying wave speed: f(x,y) = D·[1+E·sin(mπx/L)]·[1+E·sin(nπy/L)]
- Waves travel faster through regions with larger f(x,y)
- Shows beautiful refraction and focusing effects
- Complex interference patterns from variable wave speed
- Uses Neumann BCs for wave reflection at boundaries

---

# Physics PDE Simulation Analysis

## Summary

| PDE | Status | Time | Dynamics | Notes |
|-----|--------|------|----------|-------|
| gray-scott | WORKS | 30s | EXCELLENT | Beautiful spot pattern formation |
| lorenz | WORKS | 68s | EXCELLENT | Chaotic spiral/scroll waves |
| nonlinear-beams | WORKS | 248s | GOOD | Horizontal wave bands (slow) |
| burgers | WORKS | 36s | GOOD | Shock formation, diffuses at end |
| oscillators | WORKS | 45s | EXCELLENT | Beautiful spiral wave patterns |
| perona-malik | WORKS | 31s | GOOD | Edge-preserving diffusion (IC: default with noise, t_end: 1.5) |
| advecting-patterns | WORKS | 117s | EXCELLENT | Spot patterns advecting with flow |
| turing-wave | WORKS | 124s | EXCELLENT | Spot patterns with velocity field |
| growing-domains | WORKS | 100s | EXCELLENT | Spot pattern formation |
| kpz | WORKS | 38s | GOOD | Surface roughening with weak noise (nu: 0.5, lmbda: 0.2, eta: 0.1, sinusoidal IC) |
| cahn-hilliard | WORKS | 78s | EXCELLENT | Beautiful phase separation |
| superlattice | WORKS | 31s | EXCELLENT | Hexagonal spot lattice |
| kdv | WORKS | 30s | GOOD | Soliton dynamics, dispersive |
| kuramoto-sivashinsky | WORKS | 23s | EXCELLENT | Chaotic cellular patterns |
| swift-hohenberg | WORKS | 12s | EXCELLENT | Labyrinthine stripe patterns (r: 0.9, k: 0.9, t_end: 20) |
| ginzburg-landau | WORKS | 211s | EXCELLENT | Beautiful vortex lattice |

## Statistics
- **EXCELLENT**: 11 (gray-scott, lorenz, oscillators, cahn-hilliard, kuramoto-sivashinsky, ginzburg-landau, advecting-patterns, turing-wave, growing-domains, superlattice, swift-hohenberg)
- **GOOD**: 5 (burgers, nonlinear-beams, kdv, perona-malik, kpz)
- **ALL 16 PHYSICS PDEs NOW WORKING**

## Detailed Analysis

### EXCELLENT Simulations

#### gray-scott
- Classic Turing pattern formation
- Spots grow and spread from initial perturbation
- Beautiful maze-like patterns develop

#### lorenz
- Coupled reaction-diffusion system
- Develops chaotic spiral/scroll wave patterns
- Very visually interesting dynamics

#### oscillators
- Starts from noisy initial condition with velocity vectors
- Develops beautiful swirling spiral wave patterns
- Shows collective oscillator synchronization

#### cahn-hilliard
- Phase separation from noisy initial condition
- Develops clear red/blue domain structure
- Classic spinodal decomposition dynamics

#### kuramoto-sivashinsky
- Chaotic pattern formation
- Noise develops into cellular structures with well-defined features
- Very interesting dynamics

#### ginzburg-landau
- Complex field with vortex dynamics
- Noise develops into beautiful vortex lattice pattern
- Shows topological defects

#### advecting-patterns / turing-wave / growing-domains
- Pattern formation with advection
- Starts from noise, develops localized spots
- Spots move with velocity field (arrows visible)
- All three show similar excellent dynamics

#### superlattice
- Hexagonal-like superlattice pattern
- Beautiful spot array with periodic structure
- Very complex and visually interesting

### GOOD Simulations

#### burgers
- Nonlinear advection with diffusion
- Shows shock formation from initial blobs
- By frame 99 quite diffuse but structure visible

#### nonlinear-beams
- Horizontal wave band patterns
- Shows beam-like wave propagation
- Very slow (248s, 54617 steps) due to stiff dynamics

#### kdv
- Two solitons visible at start
- Solitons move and show dispersive behavior
- Dynamics relatively subtle but correct

### FIXED Simulations (Jan 6, 2026)

#### swift-hohenberg (NOW EXCELLENT)
- **Problem**: Pattern decayed to faint noise
- **Root cause**: With k=1.0, need r > k^4 = 1 for instability, but r was 0.5
- **Fix**: Set r=0.9, k=0.9 (so k^4=0.656 < r), std=0.1, t_end=20 (pattern forms by ~t=10)
- **Result**: Beautiful labyrinthine stripe patterns form from noise, stabilize by frame 50

#### perona-malik (NOW GOOD)
- **Problem**: No visible noise in initial condition to demonstrate smoothing
- **Root cause**: `type: step` IC doesn't support noise parameter
- **Fix**: Changed to `type: default` which uses preset's custom IC with 4-quadrant pattern + built-in noise
- **Result**: Shows 4 quadrants with noise, noise smoothed while edges preserved

#### kpz (NOW GOOD)
- **Problem**: Later frames appeared blank (NaN blowup)
- **Root cause**: Strong noise + nonlinear gradient_squared term caused numerical instability
- **Fix**: Reduced noise eta: 1.0->0.1, increased nu: 0.5 for stability, sinusoidal IC, dt: 0.0001
- **Result**: Shows sinusoidal smoothing with roughening from weak noise


## Summary
- All 16 physics PDEs now produce visible, interesting dynamics
- All 7 basic PDEs work correctly
- Total: 23/23 PDEs working
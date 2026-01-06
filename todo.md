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

---

# Physics PDE Comparison with VisualPDE.com (Jan 6, 2026)

## Comparison of Our Implementations vs. Real VisualPDE Equations

### MAJOR Discrepancies (Different Physics)

| PDE | Severity | Issue |
|-----|----------|-------|
| superlattice | **MAJOR** | Completely different model |
| nonlinear-beams | **MAJOR** | Different model (wave vs overdamped) |

#### superlattice
- **Our version**: Simple Swift-Hohenberg with hexagonal modulation
  - `du/dt = ε·u - (1+∇²)²u + g₂·u² - u³`
- **VisualPDE version**: 4-field coupled Brusselator + Lengyll-Epstein system
  - `du₁/dt = D_{u₁}∇²u₁ + a - (b+1)u₁ + u₁²v₁ + α·u₁u₂(u₂-u₁)`
  - `dv₁/dt = D_{u₂}∇²v₁ + b·u₁ - u₁²v₁`
  - `du₂/dt = D_{u₃}∇²u₂ + c - u₂ - 4u₂v₂/(1+u₂²) + α·u₁u₂(u₁-u₂)`
  - `dv₂/dt = D_{u₄}∇²v₂ + d[u₂ - u₂v₂/(1+u₂²)]`
- **Recommendation**: Implement full 4-field coupled system

#### nonlinear-beams
- **Our version**: 2D biharmonic wave equation
  - `du/dt = v`
  - `dv/dt = -D·∇⁴u + α·∇²(|∇u|²)`
- **VisualPDE version**: 1D overdamped beam with state-dependent stiffness
  - `dy/dt = -∂²/∂x²[E(y)·∂²y/∂x²]`
  - where `E = E* + ΔE·(1+tanh(∂²y/∂x²/ε))/2`
- **Recommendation**: Implement proper 1D overdamped beam or rename our version

### SIGNIFICANT Discrepancies (Missing Terms)

| PDE | Severity | Issue |
|-----|----------|-------|
| cahn-hilliard | **SIGNIFICANT** | Missing reaction term |
| ginzburg-landau | **SIGNIFICANT** | Simplified parameterization |
| perona-malik | **SIGNIFICANT** | Different diffusivity function |
| oscillators | **SIGNIFICANT** | Incorrect diffusion coupling |
| turing-wave | **SIGNIFICANT** | Should be hyperbolic (2nd order in time) |

#### cahn-hilliard
- **Our version**: `du/dt = M·∇²(u³ - u - γ·∇²u)`
- **VisualPDE version**: `du/dt = r·∇²(u³ - u - g·∇²u) + u - u³`
- **Missing**: Reaction term `+ u - u³`
- **Recommendation**: Add reaction term

#### ginzburg-landau
- **Our version**: `dA/dt = A + (1+ic₁)∇²A - (1+ic₃)|A|²A`
- **VisualPDE version**: `dψ/dt = (Dr+iDi)∇²ψ + (ar+iai)ψ + (br+ibi)ψ|ψ|²`
- **Missing**: Full 6-parameter version (Dr, Di, ar, ai, br, bi)
- **Recommendation**: Expand parameterization

#### perona-malik
- **Our version**: `du/dt = D·∇²u / (1 + |∇u|²/K²)`
- **VisualPDE version**: `du/dt = ∇·(e^{-D|∇u|²}·∇u)`
- **Difference**: We use rational diffusivity 1/(1+s), they use exponential e^{-s}
- **Recommendation**: Change to exponential diffusivity

#### oscillators (Van der Pol)
- **Our version**:
  - `du/dt = Du·∇²u + v`
  - `dv/dt = Dv·∇²v + μ(1-u²)v - ω²u`
- **VisualPDE version**:
  - `dX/dt = Y`
  - `dY/dt = D(∇²X + ε∇²Y) + μ(1-X²)Y - X`
- **Difference**: Diffusion coupling structure is different
- **Recommendation**: Fix diffusion term structure

#### turing-wave
- **Our version**: Standard parabolic reaction-diffusion
- **VisualPDE version**: Hyperbolic reaction-diffusion
  - `τ·∂²u/∂t² + ∂u/∂t = Du∇²u + f(u,v)`
  - `τ·∂²v/∂t² + ∂v/∂t = Dv∇²v + g(u,v)`
- **Missing**: Second-order time derivative terms (τ parameter)
- **Recommendation**: Add hyperbolic terms

### MINOR Discrepancies (Acceptable Variations)

| PDE | Issue | Status |
|-----|-------|--------|
| swift-hohenberg | Missing quintic term (c·u⁵) | OK - optional |
| gray-scott | Different notation (u/v swapped) | OK - same physics |
| lorenz | Uses D_x, D_y, D_z vs single D | OK - more flexible |
| burgers | Same formulation | ✓ Matches |
| kuramoto-sivashinsky | Has 0.5 and ν coefficients | OK - parameterized |
| kdv | Different parameterization | OK - same physics |
| advecting-patterns | Similar approach | OK |
| growing-domains | Different kinetics model | OK - valid alternative |

### Summary Table

| Category | PDEs | Count |
|----------|------|-------|
| **MAJOR** (needs rewrite) | superlattice, nonlinear-beams | 2 |
| **SIGNIFICANT** (needs update) | cahn-hilliard, ginzburg-landau, perona-malik, oscillators, turing-wave | 5 |
| **MINOR/OK** | gray-scott, lorenz, burgers, kuramoto-sivashinsky, kdv, swift-hohenberg, advecting-patterns, growing-domains, kpz | 9 |

## Priority for Updates

1. **High Priority** (Major discrepancies):
   - [ ] superlattice - implement 4-field coupled system
   - [ ] nonlinear-beams - decide: rename or reimplement

2. **Medium Priority** (Significant discrepancies):
   - [ ] cahn-hilliard - add `+ u - u³` reaction term
   - [ ] turing-wave - add hyperbolic terms (τ parameter)
   - [ ] ginzburg-landau - expand to 6-parameter version
   - [ ] perona-malik - change to exponential diffusivity
   - [ ] oscillators - fix diffusion coupling

3. **Low Priority** (Optional enhancements):
   - [ ] swift-hohenberg - add quintic term option

---

# Biology PDE Simulation Analysis

## Summary

| PDE | Status | Dynamics | Notes |
|-----|--------|----------|-------|
| allen-cahn | NEEDS FIX | BAD | Noise diffuses to uniform too fast |
| allen-cahn-standard | WORKS | GOOD | Nice phase separation dynamics |
| bacteria-flow | WORKS | EXCELLENT | Beautiful chemotaxis with flow field |
| brusselator | WORKS | EXCELLENT | Beautiful Turing spot patterns with velocity |
| cross-diffusion | NEEDS FIX | BAD | Goes to uniform green quickly |
| cyclic-competition | WORKS | EXCELLENT | Beautiful domain competition patterns |
| fisher-kpp | WORKS | EXCELLENT | Classic front propagation dynamics |
| fitzhugh-nagumo | NEEDS FIX | BAD | Equilibrates to uniform too fast |
| gierer-meinhardt | NEEDS FIX | POOR | No Turing spots, goes to uniform gradient |
| harsh-environment | WORKS | GOOD | Allee effect dynamics (ends uniform) |
| heterogeneous | WORKS | GOOD | Shows spatial heterogeneity influence |
| immunotherapy | NEEDS FIX | POOR | Tumor shrinks too fast, then all black |
| keller-segel | WORKS | GOOD | Chemotactic aggregation dynamics |
| schnakenberg | WORKS | EXCELLENT | Beautiful Turing spots with flow field |
| sir | NEEDS FIX | BAD | No epidemic wave visible, uniform → black |
| topography | WORKS | GOOD | Ring structures (but ends saturated white) |
| turing-conditions | NEEDS FIX | BAD | Doesn't show Turing patterns, goes uniform |
| vegetation | ERROR | FAILED | Dn=0.0001 below minimum 0.001 |

## Statistics
- **EXCELLENT**: 5 (bacteria-flow, brusselator, cyclic-competition, fisher-kpp, schnakenberg)
- **GOOD**: 5 (allen-cahn-standard, harsh-environment, heterogeneous, keller-segel, topography)
- **NEEDS FIX**: 7 (allen-cahn, cross-diffusion, fitzhugh-nagumo, gierer-meinhardt, immunotherapy, sir, turing-conditions)
- **ERROR**: 1 (vegetation - parameter validation error)

## Working: 10/18 (55%)

## Detailed Analysis

### EXCELLENT Simulations

#### bacteria-flow
- Localized blob with flow field grows and spreads
- Shows beautiful chemotaxis dynamics
- Velocity field visualization with arrows

#### brusselator
- Starts with noise + velocity field
- Develops beautiful localized spots by frame 50
- Classic Turing pattern formation

#### cyclic-competition
- Rock-paper-scissors dynamics
- Noise → patches → domain competition patterns
- Beautiful boundary dynamics

#### fisher-kpp
- Two initial blobs grow and spread as fronts
- Classic population wave dynamics
- By frame 99 nearly fills domain

#### schnakenberg
- Uniform → gradient → beautiful Turing spot patterns
- Shows flow field with velocity arrows
- Classic activator-inhibitor system

### GOOD Simulations

#### allen-cahn-standard
- Noise → smoothing → clear blue/red phase domains
- Shows bistable phase separation
- Good dynamics throughout

#### harsh-environment
- Blobs grow and merge with Allee effect
- Middle frames show nice dynamics
- Ends uniform yellow (t_end too long?)

#### heterogeneous
- Blobs grow with different rates
- Shows spatial heterogeneity influence
- Some low regions persist

#### keller-segel
- Multiple blobs with chemotactic gradients
- Shows aggregation dynamics
- Forms concentrated blob by frame 99

#### topography
- Ring structures around "mountains"
- Beautiful patterns at frame 50
- Ends saturated white (t_end too long)

### NEEDS FIX

#### allen-cahn
- Noise diffuses to uniform too quickly
- Need slower diffusion or different parameters

#### cross-diffusion
- Goes to uniform green by frame 50
- No pattern formation visible

#### fitzhugh-nagumo
- Square stimulus equilibrates to uniform blue
- Need more excitable parameters or spiral-inducing IC

#### gierer-meinhardt
- Should show Turing spots but goes uniform
- Reaction overwhelms pattern formation
- Need parameter tuning for Turing instability

#### immunotherapy
- Tumor shrinks too fast, all black by frame 50
- Need slower dynamics or shorter t_end

#### sir
- Uniform red → black, no wave dynamics
- Need localized infection IC for epidemic wave

#### turing-conditions
- Should demonstrate Turing instability but goes uniform
- Parameters not in Turing regime

### ERROR

#### vegetation
- **Error**: `Dn must be >= 0.001, got 0.0001`
- Config has Dn=0.0001 which is below preset minimum
- **Fix**: Increase Dn to >= 0.001

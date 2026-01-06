# Basic PDE Simulation Analysis - FINAL

## Summary

All 7 basic PDEs now work with good visible dynamics over 100 frames.

| PDE | Status | Time | Dynamics | Fixes Applied |
|-----|--------|------|----------|---------------|
| heat | WORKS | 22s | GOOD | D: 0.1 -> 0.05 |
| wave | WORKS | 55s | GOOD | (none needed) |
| advection | WORKS | 38s | GOOD | D: 0.01 -> 0.005 |
| inhomogeneous-heat | WORKS | 22s | GOOD | D: 0.1 -> 0.05 |
| schrodinger | WORKS | 75s | EXCELLENT | (none needed) |
| plate | WORKS | 75s | EXCELLENT | (none needed) |
| inhomogeneous-wave | WORKS | 16s | GOOD | gamma: 1.0->0.15, source: 0.1->0.0, t_end: 8.0->4.0 |

## Config Changes Applied

```yaml
# configs/defaults/basic/heat.yaml
parameters:
  D: 0.05  # was 0.1

# configs/defaults/basic/advection.yaml
parameters:
  D: 0.005  # was 0.01

# configs/defaults/basic/inhomogeneous_heat.yaml
parameters:
  D: 0.05  # was 0.1

# configs/defaults/basic/inhomogeneous_wave.yaml
parameters:
  gamma: 0.15  # was 1.0
  source: 0.0  # was 0.1
t_end: 4.0  # was 8.0
```

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
- Similar to heat but with source term
- Source term helps maintain visible structure even at frame 99
- Good parameter balance
- We should convert to the actual inhomogenous case: https://visualpde.com/basic-pdes/inhomogeneous-heat-equation

### schrodinger
- Wave packet with momentum creates interference patterns
- Most visually complex and interesting
- Beautiful quantum dynamics

### plate
- Constant initial condition with Dirichlet BCs
- Creates beautiful grid-like standing wave patterns
- Excellent demonstration of plate vibration modes

### inhomogeneous-wave
- Damped wave equation (no source to avoid saturation)
- Shows wave propagation with gentle damping
- Wave interference visible throughout all 100 frames
- Similar to heat - we should use the real one: https://visualpde.com/basic-pdes/inhomogeneous-wave-equation

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
| perona-malik | NEEDS FIX | 35s | BAD | Edge smoothing fails - becomes plain gradient |
| advecting-patterns | WORKS | 117s | EXCELLENT | Spot patterns advecting with flow |
| turing-wave | WORKS | 124s | EXCELLENT | Spot patterns with velocity field |
| growing-domains | WORKS | 100s | EXCELLENT | Spot pattern formation |
| kpz | NEEDS FIX | 48s | BAD | No visible roughening dynamics |
| cahn-hilliard | WORKS | 78s | EXCELLENT | Beautiful phase separation |
| superlattice | WORKS | 31s | EXCELLENT | Hexagonal spot lattice |
| kdv | WORKS | 30s | GOOD | Soliton dynamics, dispersive |
| kuramoto-sivashinsky | WORKS | 23s | EXCELLENT | Chaotic cellular patterns |
| swift-hohenberg | NEEDS FIX | 16s | BAD | Pattern decays to faint noise |
| ginzburg-landau | WORKS | 211s | EXCELLENT | Beautiful vortex lattice |

## Statistics
- **EXCELLENT**: 10 (gray-scott, lorenz, oscillators, cahn-hilliard, kuramoto-sivashinsky, ginzburg-landau, advecting-patterns, turing-wave, growing-domains, superlattice)
- **GOOD**: 3 (burgers, nonlinear-beams, kdv)
- **NEEDS FIX**: 3 (perona-malik, kpz, swift-hohenberg)

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

### NEEDS FIX Simulations

#### perona-malik
- **Problem**: Frame 0 is half black/half white step function, frame 99 is smooth gradient
- **Expected**: Edge-preserving diffusion should maintain sharp edge
- **Fix needed**: Check K parameter, may need different diffusion function or shorter t_end

#### kpz
- **Problem**: Frame 0 is uniform purple (flat), frame 99 is noisy green/yellow
- **Expected**: Should show interface roughening with visible height field dynamics
- **Fix needed**: Stochastic term may not be strong enough, or visualization may need adjustment

#### swift-hohenberg
- **Problem**: Frame 0 is high-amplitude noise, frame 99 is very faint noise (almost uniform)
- **Expected**: Should show stripe or spot pattern formation
- **Fix needed**: Check r parameter (should be positive for instability), may need higher r or different IC

## Recommendations

### Priority Fixes
1. **swift-hohenberg**: Increase `r` parameter or check if sign is correct
2. **perona-malik**: Reduce `K` or `t_end` to preserve edges
3. **kpz**: Increase noise strength or adjust visualization

### Good as-is
- 13 out of 16 physics PDEs work well with current defaults
- Most produce visually interesting dynamics suitable for training data
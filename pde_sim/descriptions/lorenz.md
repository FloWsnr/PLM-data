# Diffusively Coupled Lorenz System

A spatially extended version of the famous Lorenz equations, exhibiting spatiotemporal chaos through the interplay of local chaotic dynamics and diffusive coupling.

## Description

The Lorenz system, derived by Edward Lorenz in 1963 while studying atmospheric convection, is perhaps the most famous example of deterministic chaos. The original three-variable ODE system exhibits the iconic "butterfly" strange attractor and gave rise to the concept of sensitive dependence on initial conditions - the "butterfly effect."

The diffusively coupled version places a copy of the Lorenz oscillator at every point in space and couples neighboring oscillators through diffusion. This creates a reaction-diffusion system where:
- **Local dynamics**: Each spatial point undergoes chaotic Lorenz dynamics
- **Spatial coupling**: Diffusion smooths differences between neighboring regions

The interplay between these mechanisms produces rich spatiotemporal behavior:
- For **weak coupling** (small D): Spatially fragmented chaos with localized oscillating patches
- For **strong coupling** (large D): Synchronized large-scale oscillations approaching uniform behavior
- For **intermediate coupling**: Complex spatiotemporal patterns with coherent structures

This model demonstrates how local chaos can be spatially organized through diffusion, relevant to understanding turbulence, pattern formation in chaotic systems, and synchronization in coupled oscillator networks.

## Equations

The diffusively coupled Lorenz system:

$$\frac{\partial X}{\partial t} = D \nabla^2 X + \sigma(Y - X)$$

$$\frac{\partial Y}{\partial t} = D \nabla^2 Y + X(\rho - Z) - Y$$

$$\frac{\partial Z}{\partial t} = D \nabla^2 Z + XY - \beta Z$$

Where:
- $(X, Y, Z)$ are the Lorenz variables at each spatial point
- $\sigma$ (sigma) is the Prandtl number
- $\rho$ (rho) is the Rayleigh number (normalized)
- $\beta$ (beta) is a geometric factor
- $D$ is the diffusion coefficient (spatial coupling strength)

**Physical interpretation** (atmospheric convection):
- $X$ proportional to convective circulation intensity
- $Y$ proportional to temperature difference (ascending vs descending)
- $Z$ proportional to vertical temperature profile deviation from linear

**Classical Lorenz attractor** (ODE, D=0):
With $\sigma=10$, $\rho=28$, $\beta=8/3$, trajectories spiral around two lobes of the butterfly attractor, switching between them chaotically.

## Default Config

```yaml
solver: euler
dt: 0.00025
dx: 0.35
domain_size: 100 (inferred)

boundary_x: periodic
boundary_y: periodic

parameters:
  sigma: 10    # Prandtl number (fixed)
  rho: 30      # Rayleigh number (near classical value)
  beta: 8/3    # geometric factor (2.667)
  D: 0.5       # diffusion coefficient (coupling strength)
```

## Parameter Variants

### Lorenz (Standard)
Diffusively coupled Lorenz system:
- `sigma = 10`, `rho = 30`, `beta = 8/3`
- `D = 0.5` (intermediate coupling)
- Random initial condition: $X = 0.3 \cdot \text{RANDN} + 1$, $Z = 29$
- 3D surface plot visualization
- Embossed rendering for structure visibility

### Coupling Strength Effects

| D Value | Behavior |
|---------|----------|
| D < 0.2 | Fragmented chaos, localized patches |
| D ~ 0.5 | Complex spatiotemporal patterns |
| D > 2 | Synchronized oscillations |
| D > 5 | Nearly uniform large-wavelength oscillations |

### Exploring the System

- Start with uniform initial condition ($X(t=0) = 0$) - system remains at unstable equilibrium
- Click to perturb - initiates spreading chaotic oscillations
- Multiple clicks create interacting wave fronts
- Watch for coalescence and merging of structures over long times
- Vary $D$ to observe transition from fragmented to synchronized dynamics

### Connection to Chaos Theory

The Lorenz system exhibits:
- **Strange attractor**: Fractal structure in phase space
- **Sensitive dependence**: Exponential divergence of nearby trajectories
- **Lyapunov exponents**: One positive (chaos), one zero (along trajectory), one negative (contraction)
- **Topological mixing**: Any region eventually spreads throughout attractor

## References

- Lorenz, E.N. (1963). "Deterministic nonperiodic flow" - J. Atmos. Sci. 20:130
- Sparrow, C. (1982). "The Lorenz Equations: Bifurcations, Chaos, and Strange Attractors" - Springer
- Cross, M.C. & Hohenberg, P.C. (1993). "Pattern formation outside of equilibrium" - Rev. Mod. Phys. 65:851

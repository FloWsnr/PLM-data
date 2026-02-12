# Kuramoto-Sivashinsky Equation

A fourth-order PDE exhibiting spatiotemporal chaos - one of the simplest equations known to produce turbulent-like dynamics with characteristic coherent wavelengths.

## Description

The Kuramoto-Sivashinsky (KS) equation was independently derived in the late 1970s by Yoshiki Kuramoto (studying phase turbulence in chemical oscillations) and Gregory Sivashinsky (analyzing flame front instabilities). It has become the canonical example of a simple PDE that generates spatiotemporal chaos.

The equation arises in several physical contexts:
- **Flame fronts**: Describes the wrinkling and cellular instabilities of laminar flames
- **Thin film flows**: Models the instability of liquid films flowing down inclined planes
- **Chemical oscillations**: Phase dynamics in reaction-diffusion systems
- **Plasma physics**: Edge turbulence in tokamaks

Key mathematical features:
- **Negative diffusion** ($-\nabla^2 u$): Creates short-wavelength instability (energy injection)
- **Hyperdiffusion** ($-\nabla^4 u$): Provides large-wavenumber damping (energy dissipation)
- **Nonlinearity** ($-|\nabla u|^2$): Transfers energy between scales

The balance between destabilizing and stabilizing mechanisms produces chaos with a characteristic wavelength - the patterns are irregular but not random. The KS equation has a finite-dimensional global attractor, meaning the infinite-dimensional dynamics effectively reduce to a finite number of "active" modes.

## Equations

The standard Kuramoto-Sivashinsky equation:

$$\frac{\partial u}{\partial t} = -\nabla^2 u - \nabla^4 u - |\nabla u|^2$$

Using the product rule and introducing an auxiliary variable $v = \nabla^2 u$, this is reformulated as:

$$\frac{\partial u}{\partial t} = -\nabla \cdot \left[ (1+u) \nabla u + \nabla v \right] + u v - a u$$

with the algebraic constraint:
$$v = \nabla^2 u$$

Where:
- $u$ represents the perturbation from a flat state (e.g., flame height)
- $-\nabla^2 u$ is the destabilizing negative diffusion
- $-\nabla^4 u$ is the stabilizing fourth-order dissipation
- $-|\nabla u|^2$ is the nonlinear advection/steepening term
- $a$ is a small damping coefficient (numerical stabilization)

## Default Config

```yaml
solver: scipy    # Implicit solver for stiff 4th-order equation
adaptive: true
dt: 0.1          # Adaptive solver handles larger initial dt
resolution: 64   # ~2.34 dx (adaptive solver tolerates coarser mesh)
domain_size: 150

bc:
  x: periodic
  y: periodic

parameters:
  a: 0.03   # damping coefficient (for numerical stability)

init:
  type: gaussian-blob
  params:
    num_blobs: 5
    positions: random
    amplitude: 0.1
    width: 0.1
    background: 0.0
    aspect_ratio: 1.0
```

**Note**: Visual-PDE uses explicit solver with `dt=0.001, dx=0.5`. Our implementation uses adaptive implicit solver which is more efficient for this stiff 4th-order equation.

## Parameter Variants

### Standard K-S Chaos
Two-dimensional spatiotemporal chaos:
- `domain_size = 150`
- `a = 0.03` (slight damping for numerical stability)
- Initial condition: random Gaussian blobs around zero background
- Produces characteristic cellular/turbulent structures

### No Damping (a=0)
True Kuramoto-Sivashinsky without damping term:
- More numerically challenging
- Slightly different attractor properties
- Use with caution - may require smaller timesteps

### Strong Damping (a=0.1)
Reduced chaos:
- More regular patterns
- Faster approach to attractor
- Good for studying transition to chaos

### Chaos Characteristics

The KS equation exhibits:
- **Coherent wavelengths**: Despite chaos, patterns have characteristic size
- **Period doubling route to chaos**: As domain size increases
- **Finite-dimensional attractor**: Dimension scales with domain length
- **Sensitive dependence**: Butterfly effect in initial conditions

### Domain Size Effects

| Domain Length L | Typical Behavior |
|-----------------|------------------|
| L < 10 | Trivial/steady solutions |
| L ~ 20 | Periodic or quasiperiodic |
| L ~ 30-50 | Period-doubling bifurcations |
| L > 60 | Full spatiotemporal chaos |

## References

- Kuramoto, Y. (1978). "Diffusion-Induced Chaos in Reaction Systems" - Prog. Theor. Phys. Suppl. 64:346
- Sivashinsky, G.I. (1977). "Nonlinear analysis of hydrodynamic instability in laminar flames" - Acta Astronautica 4:1177
- Manneville, P. (1985). "Liapounov exponents for the Kuramoto-Sivashinsky model" - Macroscopic Modelling of Turbulent Flows
- Cvitanovic, P. et al. (2009). "State space geometry of chaotic Kuramoto-Sivashinsky flow" - arXiv:0709.2944

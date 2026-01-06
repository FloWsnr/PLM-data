# Hyperbolic Brusselator (Turing Wave Instability)

A modified reaction-diffusion system with inertial effects that exhibits oscillating spatial patterns - Turing instabilities with complex eigenvalues leading to wave-like behavior.

## Description

Classical (parabolic) reaction-diffusion systems can only produce Turing instabilities with real growth rates, leading to stationary spatial patterns. The hyperbolic modification introduces a finite relaxation time (inertia), enabling a fundamentally new class of instabilities: **Turing wave instabilities** with complex eigenvalues.

In Turing wave instabilities:
- Spatial modes are unstable (as in standard Turing patterns)
- The unstable modes also oscillate in time
- The result is traveling or standing wave patterns

This occurs because the hyperbolic terms introduce a delay between cause and effect - the system "remembers" its previous state. When combined with the spatial instability from differential diffusion, this memory creates oscillatory behavior.

Physical systems exhibiting such behavior include:
- **Reaction-diffusion with finite transport speeds**: Cattaneo-type heat conduction
- **Population dynamics with delay**: Generation time effects
- **Neural tissue**: Axonal propagation delays
- **Chemical reactions with intermediate steps**: Effective memory

The key parameter $\tau$ controls the "inertia" - larger $\tau$ means stronger memory effects and more pronounced wave behavior.

## Equations

The hyperbolic Brusselator:

$$\tau \frac{\partial^2 u}{\partial t^2} + \frac{\partial u}{\partial t} = D_u \nabla^2 u + a - (b+1)u + u^2 v$$

$$\tau \frac{\partial^2 v}{\partial t^2} + \frac{\partial v}{\partial t} = D_v \nabla^2 v + bu - u^2 v$$

This is implemented as a first-order system with auxiliary variables $(w, q)$:

$$\frac{\partial u}{\partial t} = w$$

$$\frac{\partial v}{\partial t} = q$$

$$\frac{\partial w}{\partial t} = \frac{D_u}{\tau} \nabla^2 u + \varepsilon \nabla^2 w + \frac{1}{\tau}\left( a + u^2 v - (b+1)u - w \right)$$

$$\frac{\partial q}{\partial t} = \frac{D_v}{\tau} \nabla^2 v + \varepsilon \nabla^2 q + \frac{1}{\tau}\left( bu - u^2 v - q \right)$$

Where:
- $(u, v)$ are the Brusselator concentrations
- $(w, q)$ are the time derivatives (velocities)
- $\tau$ is the relaxation time (memory/inertia parameter)
- $(a, b)$ are the standard Brusselator kinetic parameters
- $(D_u, D_v)$ are diffusion coefficients
- $\varepsilon$ is a small numerical diffusion for the velocity variables

**Standard Turing**: Requires $D_u < D_v$ (inhibitor diffuses faster)
**Turing wave**: Can occur with $D_u > D_v$ (activator diffuses faster) when $\tau > 0$

## Default Config

```yaml
solver: euler
dt: 0.00005
dx: 0.01 (1D) / 0.02 (2D)
domain_size: 0.5 (1D) / 10 (2D)
dimension: 1 or 2

boundary_x: neumann
boundary_y: neumann

parameters:
  a: 5           # Brusselator feed rate
  b: 9           # Brusselator removal rate
  tau: 1         # [0, 1] - relaxation time (key parameter)
  D_v: 8         # [7, 9] - inhibitor diffusion
  epsilon: 0.001 # numerical diffusion
```

## Parameter Variants

### BrusselatorTuringWave1D
One-dimensional Turing wave:
- `domain_size = 0.5`
- `D = 2` (D_u = D via preset), `D_v = 8`
- `tau = 1`
- Initial condition: Cosine perturbation
- Shows oscillating cosine mode (linear theory prediction)

### BrusselatorTuringWave2D
Two-dimensional Turing wave:
- `domain_size = 10`
- Same parameters, extended to 2D
- Random initial perturbations
- Exhibits complex transient dynamics converging to wave-like patterns

### TuringWaveFHN / TuringWaveFHN2D
FitzHugh-Nagumo variant:
- Different kinetics: $f(u,v) = u - au^3 + v - b$
- Similar hyperbolic structure
- Parameters: `delta = 1.1`, `a = 0.32`, `b = 5.0`

### Effect of tau (Relaxation Time)

| tau Value | Behavior |
|-----------|----------|
| tau -> 0 | Standard parabolic Brusselator |
| tau ~ 0.1 | Weak oscillations, decaying |
| tau ~ 0.5 | Intermediate - transition regime |
| tau ~ 1 | Strong Turing wave instability |
| tau > 1 | Larger oscillation amplitude |

### Diffusion Ratio Effects

With $D_u = D$ and $D_v$ fixed:
- Decreasing $D$ (or increasing tau) reduces the instability
- The amplitude of oscillating patterns decreases
- Eventually patterns become purely decaying

### Linear Theory Prediction

For small perturbations $\propto e^{ikx + \lambda t}$:
- Standard Turing: $\lambda$ is real (growing/decaying stationary modes)
- Turing wave: $\lambda = \sigma \pm i\omega$ (growing oscillating modes)

The frequency $\omega$ depends on $\tau$ and the kinetic parameters.

## References

- Zemskov, E.P. et al. (2022). "Turing and wave instabilities in hyperbolic reaction-diffusion systems" - arXiv:2204.13820
- Mendez, V. & Fedotov, S. (2010). "Reaction-transport systems: mesoscopic foundations, fronts, and spatial instabilities" - Springer
- Fort, J. & Mendez, V. (2002). "Wavefronts in time-delayed reaction-diffusion systems" - Phys. Rev. Lett. 89:178101

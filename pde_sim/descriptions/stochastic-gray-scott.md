# Stochastic Gray-Scott Model

A noise-driven reaction-diffusion system demonstrating stochastic resonance, where random fluctuations can induce and control pattern formation.

## Description

The stochastic Gray-Scott model extends the classical Gray-Scott reaction-diffusion system by adding spatiotemporal noise. This creates a system where randomness is not just a perturbation but can fundamentally alter the dynamics through **stochastic resonance** - the counterintuitive phenomenon where noise can enhance rather than destroy ordered behavior.

Key phenomena in stochastic reaction-diffusion systems:
- **Noise-induced patterns**: Patterns emerge that would not exist in the deterministic system
- **Stochastic resonance**: Intermediate noise levels optimize pattern formation
- **Stochastic extinction**: Very strong noise destroys all patterns
- **Fluctuation-driven transitions**: Random kicks can push the system between different states

In the stochastic Gray-Scott model:
- At zero noise ($\sigma = 0$): The deterministic system may have a stable homogeneous state
- At small noise ($\sigma \sim 0.1-0.2$): Noise destabilizes the homogeneous state, inducing Turing-like patterns (stripes, spots)
- At large noise ($\sigma > 0.3$): Noise destroys patterns entirely (stochastic extinction)

This demonstrates that biological pattern formation may rely on noise - cells may use stochastic fluctuations to robustly generate spatial organization even when deterministic mechanisms would fail.

## Equations

The stochastic Gray-Scott equations:

$$\frac{\partial u}{\partial t} = \nabla^2 u + u^2 v - (a+b)u + \sigma \frac{dW_t}{dt} u$$

$$\frac{\partial v}{\partial t} = D \nabla^2 v - u^2 v + a(1-v)$$

Where:
- $(u, v)$ are the chemical concentrations (same as deterministic Gray-Scott)
- $a$ is the feed rate
- $b$ is the kill/removal rate
- $D = 4$ is the diffusion ratio
- $\sigma$ is the noise intensity
- $W_t$ is an approximation of a Brownian sheet (spatiotemporal white noise)
- The noise is multiplicative (multiplied by $u$)

**Noise implementation**:
The white noise term is scaled as:
$$\frac{dW_t}{dt} \propto \frac{1}{\sqrt{\Delta t \cdot \Delta x^N}} \xi(t, \mathbf{x})$$

where $\xi$ is a standard normal random variable independent at each grid point and timestep, and $N$ is the spatial dimension.

## Default Config

```yaml
solver: euler  # IMPORTANT: Only Euler scheme works correctly for stochastic terms
dt: 0.02
dx: 0.6
domain_size: 300

boundary_x: periodic
boundary_y: periodic

parameters:
  a: 0.037     # [0, 0.1] - feed rate
  b: 0.04      # [0.04, 0.1] - kill rate
  sigma: 0.000 # [0, 0.8] - noise intensity (start at 0)
  D: 4         # diffusion ratio (inherited from GrayScott parent)
```

## Parameter Variants

### StochasticGrayScott (Standard)
Stochastic pattern formation demonstration:
- `a = 0.037`, `b = 0.04` (parameters where deterministic system is stable)
- `D = 4` (diffusion ratio)
- `sigma = 0` initially (deterministic - increase to add noise)
- `domain_size = 300`, `dx = 0.6`
- Initial condition: Small circular patch of $u$

### Noise Intensity Regimes

| sigma | Behavior |
|-------|----------|
| sigma = 0 | Deterministic: stable homogeneous state |
| 0 < sigma < 0.1 | Weak noise: occasional fluctuations |
| 0.1 < sigma < 0.2 | Stochastic resonance: noise induces patterns (stripes/spots) |
| 0.2 < sigma < 0.3 | Strong patterns, transitioning to disorder |
| sigma > 0.3 | Stochastic extinction: patterns destroyed |

### Exploration Protocol

1. Start with `sigma = 0` - observe stable homogeneous state
2. Click to perturb - perturbations decay back to uniform
3. Increase `sigma` to 0.1-0.15 - observe pattern emergence
4. Fine-tune `sigma` - smaller values favor stripes, larger values favor spots
5. Increase further to ~0.3 - observe pattern destruction

### Important Numerical Note

**The stochastic terms only work correctly with forward Euler timestepping!**

Other timestepping schemes (RK4, midpoint, etc.) do not properly scale with the timestep for stochastic terms. The noise scaling $\propto 1/\sqrt{\Delta t}$ is built into the Euler step.

### Physical/Biological Relevance

Stochastic pattern formation is relevant to:
- **Morphogenesis**: Cells use noise to robustly generate body plans
- **Gene regulation**: Stochastic gene expression can create spatial patterns
- **Chemical reactions**: Low molecule counts make deterministic models invalid
- **Ecological patterns**: Environmental fluctuations drive vegetation patterns

## References

- Biancalani, T. et al. (2010). "Stochastic Turing patterns in the Brusselator model" - Phys. Rev. E 81:046215
- McKane, A.J. & Newman, T.J. (2005). "Predator-prey cycles from resonant amplification of demographic stochasticity" - Phys. Rev. Lett. 94:218102
- Woolley, T.E. et al. (2011). "Stochastic reaction and diffusion on growing domains" - Phys. Rev. E 84:046216
- Horsthemke, W. & Lefever, R. (1984). "Noise-Induced Transitions" - Springer

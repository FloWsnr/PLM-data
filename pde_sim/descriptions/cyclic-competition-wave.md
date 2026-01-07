# Cyclic Competition Wave

A variant of the cyclic competition (rock-paper-scissors) model with equal diffusion coefficients, demonstrating wave-induced spatiotemporal chaos without Turing instability.

## Description

This model differs from the standard cyclic competition preset in two key ways:

1. **Equal diffusion**: All three species have the same diffusion coefficient, preventing Turing pattern formation
2. **Edge-localized initial conditions**: Species start only in the left 10% of the domain, creating an invasion front

The result is spatiotemporal chaos driven by wave interactions rather than Turing instability. This demonstrates that complex dynamics can emerge from simple reaction-diffusion systems even without differential diffusion.

### Key Physical Behaviors

**Invasion waves**: Species expand from the initial domain edge as traveling waves. The cyclic competition creates rotating wave fronts.

**Spatiotemporal chaos**: Wave collisions and interactions lead to chaotic, unpredictable dynamics despite deterministic equations.

**No Turing patterns**: With equal diffusion, the homogeneous steady state is stable to small perturbations. Patterns arise purely from the initial spatial structure and wave dynamics.

**Biodiversity maintenance**: Despite chaos, all three species coexist long-term due to the cyclic dominance relationship.

## Equations

$$\frac{\partial u}{\partial t} = D \nabla^2 u + u(1 - u - av - bw)$$

$$\frac{\partial v}{\partial t} = D \nabla^2 v + v(1 - bu - v - aw)$$

$$\frac{\partial w}{\partial t} = D \nabla^2 w + w(1 - au - bv - w)$$

where:
- $u$, $v$, $w$ are species densities
- $D$ is the common diffusion coefficient (equal for all species)
- $a < 1$ is the weak competition coefficient
- $b > 1$ is the strong competition coefficient

The cyclic dominance: u beats v, v beats w, w beats u.

## Default Config

```yaml
preset: cyclic-competition-wave
solver: euler
dt: 0.01
domain_size: 200

bc:
  x: periodic
  y: periodic

parameters:
  a: 0.8      # weak competition
  b: 1.9      # strong competition
  D: 0.3      # equal diffusion for all species

init:
  type: default  # Edge-localized (left 10% of domain)
  params:
    edge_fraction: 0.1
    noise: 0.001
```

## Parameter Variants

### Standard Wave
- `D = 0.3`: Moderate wave speed
- Creates complex spatiotemporal chaos

### Fast Waves
- `D = 1.0`: Higher wave speed
- Faster invasion but similar chaos

### Slow Waves
- `D = 0.1`: Lower wave speed
- More localized dynamics

## Comparison with cyclic-competition

| Feature | cyclic-competition | cyclic-competition-wave |
|---------|-------------------|------------------------|
| Diffusion | Du=2.0, Dv=Dw=0.5 | Du=Dv=Dw=0.3 |
| Initial conditions | Central sech bump | Left edge localization |
| Pattern mechanism | Turing instability | Wave dynamics |
| Behavior | Spiral waves | Spatiotemporal chaos |

## References

- May, R. M., & Leonard, W. J. (1975). Nonlinear aspects of competition between three species. SIAM Journal on Applied Mathematics, 29(2), 243-253.
- Reichenbach, T., Mobilia, M., & Frey, E. (2007). Mobility promotes and jeopardizes biodiversity in rock-paper-scissors games. Nature, 448(7157), 1046-1049.
- VisualPDE cyclicCompetitionWave preset

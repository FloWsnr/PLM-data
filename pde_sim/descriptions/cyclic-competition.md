# Cyclic Competition Model

A three-species reaction-diffusion system modeling rock-paper-scissors ecological dynamics, exhibiting spiral waves and complex spatiotemporal chaos.

## Description

The cyclic competition model, also known as the spatial rock-paper-scissors game, describes ecosystems where three species compete in a non-transitive (cyclic) manner: species A outcompetes B, B outcompetes C, and C outcompetes A. This creates an endless cycle of competition with no single winner.

This is a generalized Lotka-Volterra system extended to three competing populations with spatial diffusion. When parameters satisfy $a < 1 < b$, each species outcompetes exactly one other species while being outcompeted by the third.

Key phenomena include:
- **Spiral waves**: The dominant feature of the system - rotating spiral waves that form spontaneously from structured initial conditions
- **Biodiversity maintenance**: All three species coexist indefinitely despite competitive exclusion in well-mixed systems
- **Wave-induced chaos**: Complex spatiotemporal dynamics even without Turing-like instabilities
- **Critical mobility threshold**: Above a certain diffusion rate, biodiversity can be lost

The model has been studied in the context of:
- Microbial ecosystems (e.g., E. coli strains producing colicins)
- Coral reef competition
- Lizard mating strategies (side-blotched lizards)
- Evolutionary game theory
- Pattern formation in non-equilibrium systems

A key insight is that spatial structure (differential diffusion rates) can maintain biodiversity that would be lost in well-mixed populations. The entanglement of spiral waves creates refugia where each species can persist.

## Equations

$$\frac{\partial u}{\partial t} = D_u \nabla^2 u + u(1 - u - av - bw)$$

$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + v(1 - bu - v - aw)$$

$$\frac{\partial w}{\partial t} = D_w \nabla^2 w + w(1 - au - bv - w)$$

Where:
- $u, v, w$ are the population densities of the three species
- $D_u, D_v, D_w$ are diffusion coefficients
- $a < 1$: Weak competitive effect (species is outcompeted)
- $b > 1$: Strong competitive effect (species outcompetes)
- The carrying capacity for each species (alone) is 1

Competition structure (with $a < 1 < b$):
- $u$ beats $v$ (coefficient $a$), loses to $w$ (coefficient $b$)
- $v$ beats $w$ (coefficient $a$), loses to $u$ (coefficient $b$)
- $w$ beats $u$ (coefficient $a$), loses to $v$ (coefficient $b$)

## Default Config

```yaml
solver: euler
dt: 0.005
dx: 1.0
domain_size: 500

boundary_x: neumann
boundary_y: neumann

num_species: 3

parameters:
  a: 0.8    # weak competition coefficient (< 1)
  b: 1.9    # strong competition coefficient (> 1)
  D_u: 2    # species u diffusion
  D_v: 0.5  # species v diffusion
  D_w: 0.5  # species w diffusion
```

## Parameter Variants

### cyclicCompetition (Standard)
Base configuration with differential diffusion producing spiral waves.
- `a = 0.8`, `b = 1.9`: Cyclic competition structure
- `D_u = 2`, `D_v = 0.5`, `D_w = 0.5`: Asymmetric diffusion
- Initial condition: `sech(r) = 1/cosh(sqrt((x-L_x/2)^2+(y-L_y/2)^2))` - all species localized at center
- Evolves into spiral wave chaos

### cyclicCompetitionWave
Wave instability configuration without Turing-like requirements.
- Equal diffusion: `D_u = D_v = D_w = 0.3`
- Initial condition: `H(0.1-x/L_x)` - Heaviside function, species only in left 10% of domain
- Demonstrates wave-induced spatiotemporal chaos
- Shows that spiral waves persist even with equal diffusion once established
- **Not yet implemented** - requires Heaviside-based initial condition support

## Notes

- Setting all diffusion coefficients equal (e.g., `D_u = 0.5`) after spiral waves form will maintain them
- The system can exhibit spiral waves without satisfying traditional Turing conditions
- Above a critical mobility threshold, biodiversity is lost and one species dominates
- The parameter combination ensures cyclic dominance: each species has both a weak ($a < 1$) and strong ($b > 1$) competitive interaction

## References

- May, R. M., & Leonard, W. J. (1975). Nonlinear aspects of competition between three species. SIAM Journal on Applied Mathematics, 29(2), 243-253.
- Reichenbach, T., Mobilia, M., & Frey, E. (2007). Mobility promotes and jeopardizes biodiversity in rock-paper-scissors games. Nature, 448(7157), 1046-1049.
- Szolnoki, A., et al. (2014). Cyclic dominance in evolutionary games: a review. Journal of the Royal Society Interface, 11(100), 20140735.

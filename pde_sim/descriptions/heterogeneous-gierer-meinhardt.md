# Heterogeneous Gierer-Meinhardt Model

A spatially heterogeneous version of the Gierer-Meinhardt model demonstrating how spatial gradients in parameters lead to isolated patterns and moving spikes.

## Description

This model extends the classical Gierer-Meinhardt activator-inhibitor system by introducing spatial heterogeneity - parameters that vary across the domain. This breaks the translational symmetry of the system and leads to qualitatively different behaviors than the homogeneous case.

The heterogeneity is introduced through:
- $G(x)$: Spatial variation in base activator production
- $H(x)$: Spatial variation in activator decay rate

Key phenomena in heterogeneous systems:
- **Localized patterns**: Patterns form preferentially in certain regions
- **Spike pinning**: Spikes become localized at specific positions
- **Spike motion**: In 1D, spikes can slowly migrate along parameter gradients
- **Hopf instabilities**: Local oscillations in spike amplitude
- **Spike creation/annihilation**: Dynamic spike birth and death at domain boundaries

The interplay between heterogeneity and reaction-diffusion dynamics produces:
1. Position-dependent pattern wavelengths
2. Amplitude variations across the domain
3. Complex transient dynamics during pattern selection
4. Asymmetric spike profiles

This is relevant for biological systems where:
- Morphogen gradients exist (developmental biology)
- Environmental conditions vary spatially (ecology)
- Material properties are non-uniform (chemical systems)

## Equations

$$\frac{\partial u}{\partial t} = \nabla^2 u + a + G(x) + \frac{u^2}{v} - [b + H(x)]u$$

$$\frac{\partial v}{\partial t} = D \nabla^2 v + u^2 - cv$$

With heterogeneity functions:
$$G(x) = \frac{Ax}{L}, \quad H(x) = \frac{Bx}{L}$$

Where:
- $u$ is the activator concentration
- $v$ is the inhibitor concentration
- $a, b, c$ are kinetic parameters
- $D > 1$ is the inhibitor diffusion ratio
- $A$ controls production gradient
- $B$ controls decay gradient
- $L$ is the domain length

## Default Config

```yaml
solver: euler
dt: 0.0003
dx: 0.5
domain_size: 100

# Boundary conditions per species (Visual PDE style):
# u: Dirichlet (u=0 at all boundaries)
# v: Neumann (zero flux at all boundaries)
# Note: Our current implementation uses per-axis BCs which differs from Visual PDE

num_species: 2

parameters:
  a: 1
  b: 1.5
  c: 6.1
  D: 55      # inhibitor diffusion
  A: 0       # range: [-1, 1], step: 0.1, production gradient
  B: 0       # range: [0, 5], step: 0.1, decay gradient

# Initial conditions:
# Visual PDE: v=1 (uniform), u unset (0 or noise)
# Patterns emerge from Dirichlet BC on u perturbing the equilibrium
```

## Parameter Variants

### GMHeterogeneous2D
Two-dimensional heterogeneous system.
- Mixed boundary conditions: Dirichlet for u, Neumann for v
- `A = 0` (range: [-1, 1]): Production heterogeneity
- `B = 0` (range: [0, 5]): Decay heterogeneity
- Even without heterogeneity ($A = B = 0$), boundary conditions break symmetry
- Non-zero $A$ or $B$ creates position-dependent pattern sizes

### GMHeterogeneousOscillationsMixedBCs
One-dimensional system showing spike dynamics.
- Linear heterogeneity $H(x) = Bx/L$
- Demonstrates:
  - Hopf instability leading to oscillating spike amplitudes
  - Spike motion toward lower decay regions
  - Spike annihilation at boundaries
  - New spike creation as space opens up
- Long-term cyclic behavior of creation-motion-annihilation

### GMHeterogeneousOscillationsDirichletBCs
Cleaner spike dynamics with Dirichlet conditions on both species.
- $v = 2$ at boundaries dampens edge artifacts
- Removes boundary spike from previous variant
- Clearer visualization of internal spike migration
- Reducing $v$ boundary value to 1 restores boundary spikes

## Behavior Summary

| Parameter | Effect |
|-----------|--------|
| $A < 0$ | Larger patterns on right side |
| $A > 0$ | Larger patterns on left side |
| $B > 0$ | Spikes migrate leftward (toward lower decay) |
| Dirichlet BC | Boundary-induced pattern nucleation |

## Notes

- Pattern selection depends on both parameter gradients and boundary conditions
- Restarting simulations with heterogeneity can produce different transient oscillations
- The 1D dynamics provide insight into higher-dimensional behaviors
- Reducing inhibitor diffusion $D$ changes pattern wavelength
- Spike motion speed depends on gradient steepness

## References

- Iron, D., Ward, M. J., & Wei, J. (2001). The stability of spike solutions to the one-dimensional Gierer-Meinhardt model. Physica D, 150(1-2), 25-62.
- Kolokolnikov, T., Ward, M. J., & Wei, J. (2005). The existence and stability of spike equilibria in the one-dimensional Gray-Scott model. Studies in Applied Mathematics, 115(1), 21-71.
- Ward, M. J., & Wei, J. (2003). Hopf bifurcations and oscillatory instabilities of spike solutions for the one-dimensional Gierer-Meinhardt model. Journal of Nonlinear Science, 13(2), 209-264.

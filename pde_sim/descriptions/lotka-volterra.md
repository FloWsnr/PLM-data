# Lotka-Volterra Predator-Prey Model

A two-species reaction-diffusion system modeling the spatial dynamics of predator-prey interactions, exhibiting oscillatory population cycles, pursuit waves, and spiral wave patterns.

## Description

The Lotka-Volterra predator-prey model is one of the foundational mathematical models in ecology, describing the dynamics between a prey species (u) and its predator (v). Originally formulated as a system of ordinary differential equations by Alfred Lotka (1925) and Vito Volterra (1926), the spatial extension adds diffusion terms to capture how populations spread across landscapes.

The model captures several fundamental ecological phenomena:
- **Population oscillations**: In the absence of spatial effects, the ODE system produces neutral oscillations where predator and prey populations cycle out of phase
- **Pursuit waves**: When diffusion is included, predators can "chase" prey across space, creating traveling wave fronts
- **Spiral waves**: Localized perturbations can nucleate rotating spiral patterns similar to those in excitable media
- **Turing patterns**: For certain parameter combinations (especially when prey diffuse faster than predators), stationary spatial patterns can emerge

The biological interpretation is:
- Prey grow exponentially in the absence of predators (rate alpha)
- Predators consume prey proportionally to encounters (rate beta)
- Predators reproduce by converting consumed prey (efficiency delta)
- Predators die in the absence of prey (rate gamma)

Key ecological insights from the model include:
- Predator-prey systems naturally oscillate even without external forcing
- Spatial structure can stabilize otherwise unstable dynamics
- The coexistence equilibrium depends on the ratio of birth and death rates
- Differential mobility between species creates complex spatial dynamics

## Equations

$$\frac{\partial u}{\partial t} = D_u \nabla^2 u + \alpha u - \beta u v$$

$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + \delta u v - \gamma v$$

Where:
- $u$ is the prey population density
- $v$ is the predator population density
- $D_u, D_v$ are diffusion coefficients for prey and predator
- $\alpha$ is the prey growth rate (intrinsic rate of increase)
- $\beta$ is the predation rate (functional response coefficient)
- $\delta$ is the predator reproduction efficiency (numerical response)
- $\gamma$ is the predator death rate (mortality in absence of prey)

### Fixed Points

The system has two fixed points:
1. **Trivial**: $(u^*, v^*) = (0, 0)$ - both species extinct (unstable saddle)
2. **Coexistence**: $(u^*, v^*) = (\gamma/\delta, \alpha/\beta)$ - stable center (neutral stability in ODE, can be destabilized by diffusion)

### Oscillation Properties

For the ODE system (no diffusion), the period of oscillations near the coexistence equilibrium is approximately:
$$T \approx \frac{2\pi}{\sqrt{\alpha \gamma}}$$

## Default Config

```yaml
preset: lotka-volterra

parameters:
  Du: 1.0      # Prey diffusion
  Dv: 0.5      # Predator diffusion
  alpha: 1.0   # Prey growth rate
  beta: 0.1    # Predation rate
  delta: 0.075 # Predator efficiency
  gamma: 0.5   # Predator death rate

solver: euler
dt: 0.01
resolution: 128
domain_size: 100.0

bc:
  x-: neumann:0
  x+: neumann:0
  y-: neumann:0
  y+: neumann:0

init:
  type: random-uniform
  params:
    low: 5.0
    high: 10.0
```

## Parameter Variants

### Default (Pursuit Dynamics)
Base configuration showing predator-prey pursuit patterns.
- `Du = 1.0`, `Dv = 0.5`: Prey diffuse faster than predators
- `alpha = 1.0`, `gamma = 0.5`: Moderate growth and death rates
- `beta = 0.1`, `delta = 0.075`: Weak coupling promotes spatial structure
- Equilibrium: $u^* \approx 6.67$, $v^* = 10$

### Spiral Waves
Configuration promoting spiral wave formation.
- Equal or similar diffusion: `Du = 0.5`, `Dv = 0.5`
- Higher predation coupling: `beta = 0.2`, `delta = 0.15`
- Localized initial perturbation triggers rotating spirals
- Periodic boundaries help maintain spiral patterns

### Fast Oscillations
Configuration with rapid population cycling.
- Higher growth rates: `alpha = 2.0`, `gamma = 1.0`
- Faster characteristic oscillation frequency
- Good for observing multiple cycles in short simulations

### Turing-like Patterns
Configuration approaching Turing instability conditions.
- Large diffusion ratio: `Du = 2.0`, `Dv = 0.1`
- Prey diffuse much faster than predators
- Can produce stationary spatial patterns under right conditions

## Boundary Conditions

- **Neumann (no-flux)**: Populations cannot leave the domain - represents an isolated ecosystem
- **Periodic**: Represents a patch of a larger homogeneous environment
- **Dirichlet**: Can represent source/sink boundaries or habitat edges

## Notes

- Population values should remain non-negative (biological constraint)
- The coexistence equilibrium in the ODE is a center (neutrally stable), making dynamics sensitive to perturbations
- For very long simulations, numerical diffusion can cause slow decay of oscillations
- The spatial model can exhibit richer dynamics than the ODE, including spatial pattern formation
- Low population densities can lead to numerical issues; consider using implicit solvers for stiff regimes

## References

- Lotka, A. J. (1925). Elements of Physical Biology. Williams and Wilkins.
- Volterra, V. (1926). Fluctuations in the abundance of a species considered mathematically. Nature, 118, 558-560.
- Murray, J. D. (2002). Mathematical Biology I: An Introduction (3rd ed.). Springer. Chapter 3.
- Sherratt, J. A. (2001). Periodic travelling waves in cyclic predator-prey systems. Ecology Letters, 4(1), 30-37.
- Petrovskii, S. V., & Malchow, H. (2001). Wave of chaos: New mechanism of pattern formation in spatio-temporal population dynamics. Theoretical Population Biology, 59(2), 157-174.

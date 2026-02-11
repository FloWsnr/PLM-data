# Phase-Field Crystal Equation

A sixth-order conserved dynamics model for crystallization, describing the emergence of periodic lattice structures from an undercooled melt at atomic length scales but diffusive time scales.

## Description

The Phase-Field Crystal (PFC) model, introduced by Elder et al. (2002), bridges the gap between traditional phase-field models (which operate at mesoscopic scales) and molecular dynamics (which resolves individual atoms). It naturally produces periodic density modulations corresponding to crystalline lattice structures, while operating on diffusive time scales much longer than atomic vibration periods.

Key physical phenomena captured by this equation:
- **Crystallization**: Spontaneous formation of periodic lattice structures from a uniform (liquid) state when the undercooling parameter $\epsilon$ exceeds a critical value
- **Grain boundaries**: Defects between misoriented crystal grains emerge naturally
- **Elastic interactions**: Long-range elastic effects are automatically included through the crystal periodicity
- **Topological defects**: Dislocations, vacancies, and grain boundary motion arise from the dynamics
- **Coarsening**: Grain growth and Ostwald ripening of crystal domains

The model is derived by minimizing a Swift-Hohenberg-type free energy functional with conserved (diffusive) dynamics, making it a conserved variant of the Swift-Hohenberg equation. The sixth-order spatial derivatives select a preferred wavelength for the density modulations.

Physical applications include:
- **Solidification**: Modeling dendritic growth and microstructure formation
- **Thin film epitaxy**: Growth of crystalline layers with lattice mismatch
- **Colloidal crystals**: Self-assembly of colloidal particles into ordered structures
- **Polycrystalline materials**: Grain boundary networks and texture evolution

## Equations

The Phase-Field Crystal equation:

$$\frac{\partial \phi}{\partial t} = \nabla^2 \left[ (1 + \nabla^2)^2 \phi + \phi^3 - \epsilon \phi \right]$$

Expanding the $(1+\nabla^2)^2$ operator:

$$\frac{\partial \phi}{\partial t} = (1-\epsilon)\nabla^2\phi + 2\nabla^4\phi + \nabla^6\phi + \nabla^2(\phi^3)$$

This is derived from the free energy functional:

$$\mathcal{F}[\phi] = \int \left[ \frac{1}{2}\phi\left[(1+\nabla^2)^2 - \epsilon\right]\phi + \frac{1}{4}\phi^4 \right] d\mathbf{r}$$

with conserved dynamics $\partial_t \phi = \nabla^2 (\delta\mathcal{F}/\delta\phi)$.

Where:
- $\phi$ represents the local atomic density deviation from the mean
- $\epsilon$ is the undercooling parameter (distance from the melting point)
- The $(1+\nabla^2)^2$ operator selects a preferred wavelength $2\pi$ for density modulations

## Default Config

```yaml
solver: scipy
adaptive: true
dt: 0.001
resolution: [64, 64]
domain_size: [50.0, 50.0]

bc:
  x-: periodic
  x+: periodic
  y-: periodic
  y+: periodic

init:
  type: phase-field-crystal-default
  params:
    mean_density: 0.2
    amplitude: 0.05

parameters:
  epsilon: 0.325
```

## Parameter Variants

### epsilon = 0.325 (Standard quench)
Moderate undercooling producing well-defined hexagonal crystal structures in 2D. Good balance between crystallization speed and pattern quality.

### epsilon = 0.5 (Deep quench)
Strong undercooling leading to rapid crystallization with many nucleation sites. Produces smaller grains and more grain boundaries.

### epsilon = 0.1 (Shallow quench)
Near the melting point. Slow crystallization with fewer, larger crystal domains. May show coexistence of liquid and crystal phases.

### Physical Interpretation

| Parameter | Physical Meaning |
|-----------|------------------|
| $\phi > 0$ | Locally crystalline (density above mean) |
| $\phi < 0$ | Locally liquid-like (density below mean) |
| $\epsilon$ | Undercooling depth (larger = deeper into crystal phase) |
| Mean $\phi$ | Average density (controls crystal structure type) |

### Stability Diagram

- **Uniform liquid**: Stable for $\epsilon < \epsilon_c(\bar{\phi})$
- **Crystalline (stripes in 1D, hexagons in 2D)**: Stable for $\epsilon > \epsilon_c$
- **Coexistence region**: Both phases locally stable, separated by nucleation barrier

## Numerical Notes

The sixth-order spatial derivatives make this PDE very stiff. For production runs, use the `scipy` solver with `adaptive: true`. The Euler solver requires extremely small time steps and is only suitable for very short test runs on coarse grids.

## References

- Elder, K.R. et al. (2002). "Modeling Elasticity in Crystal Growth" — Phys. Rev. Lett. 88:245701
- Elder, K.R. & Grant, M. (2004). "Modeling elastic and plastic deformations in nonequilibrium processing using phase field crystals" — Phys. Rev. E 70:051605
- Emmerich, H. et al. (2012). "Phase-field-crystal models for condensed matter dynamics on atomic length and diffusive time scales" — Advances in Physics 61:665-743

# Fisher-KPP Equation

## Mathematical Formulation

The Fisher-Kolmogorov-Petrovsky-Piskunov (Fisher-KPP) equation:

$$\frac{\partial u}{\partial t} = D \nabla^2 u + ru\left(1 - \frac{u}{K}\right)$$

where:
- $u$ is the population density
- $D$ is the diffusion coefficient
- $r$ is the intrinsic growth rate
- $K$ is the carrying capacity

## Physical Background

The Fisher-KPP equation combines:
1. **Random dispersal**: Diffusion term modeling organism movement
2. **Logistic growth**: Population growth with density dependence

This is the simplest reaction-diffusion equation exhibiting **traveling wave solutions**.

## Traveling Waves

The equation admits traveling wave solutions $u(x,t) = U(x - ct)$ connecting:
- $u = 0$ (unstable, invasion front)
- $u = K$ (stable, carrying capacity)

**Minimum wave speed**: $c_{\min} = 2\sqrt{Dr}$

For initial conditions with compact support, the wave speed approaches $c_{\min}$.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Diffusion | $D$ | Dispersal rate | 0.01 - 0.5 |
| Growth rate | $r$ | Per capita growth | 0.1 - 2.0 |
| Carrying capacity | $K$ | Maximum density | 0.1 - 10 |

## Wave Profile

The traveling wave has a characteristic **sigmoidal shape**:
$$U(\xi) \approx \frac{K}{1 + e^{\lambda \xi}}$$

where $\xi = x - ct$ and $\lambda$ depends on wave speed.

## Applications

1. **Population biology**: Species invasions, range expansions
2. **Epidemiology**: Disease spread (SIS model variant)
3. **Genetics**: Advantageous allele spread
4. **Ecology**: Habitat colonization
5. **Cancer biology**: Tumor growth fronts

## Historical Notes

- R.A. Fisher (1937): Gene spread in populations
- Kolmogorov, Petrovsky, Piskunov (1937): Mathematical analysis

Both papers appeared in the same year, establishing the field of mathematical population genetics.

## Generalizations

- **Bistable variant**: Allen-Cahn equation with Allee effect
- **Multi-species**: Competitive Lotka-Volterra systems
- **Nonlocal**: Integro-differential equations with long-range dispersal

## References

- Fisher, R.A. (1937). *The Wave of Advance of Advantageous Genes*
- Murray, J.D. (2003). *Mathematical Biology I*

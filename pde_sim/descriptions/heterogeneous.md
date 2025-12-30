# Heterogeneous Environment Model

## Mathematical Formulation

Reaction-diffusion with spatially varying parameters:

$$\frac{\partial u}{\partial t} = D \nabla^2 u + r(x,y) \cdot u(1 - u)$$

where:
$$r(x,y) = r_{\text{mean}} + r_{\text{amp}} \sin(2\pi f x)$$

- $u$ is population density
- $D$ is diffusion coefficient
- $r(x,y)$ is the spatially varying growth rate
- $r_{\text{mean}}$ is mean growth rate
- $r_{\text{amp}}$ is amplitude of variation
- $f$ is spatial frequency

## Physical Background

Real environments are rarely uniform. This model explores how **spatial heterogeneity** affects population dynamics:

- **Favorable patches**: $r(x,y) > 0$, population can grow
- **Unfavorable patches**: $r(x,y) < 0$, population declines
- **Intermediate zones**: Marginal habitat

The sinusoidal variation creates alternating bands of good and bad habitat.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Diffusion | $D$ | Movement rate | 0.01 - 10 |
| Mean growth | $r_{\text{mean}}$ | Average growth rate | 0 - 5 |
| Amplitude | $r_{\text{amp}}$ | Variation strength | 0 - 2 |
| Frequency | $f$ | Patches per unit length | 1 - 10 |

## Effective Dynamics

The behavior depends on the relationship between:
- **Patch size**: $\lambda = 1/f$
- **Dispersal scale**: $\ell = \sqrt{D/r_{\text{mean}}}$

| Regime | Behavior |
|--------|----------|
| $\lambda \ll \ell$ | Averaging; feels mean environment |
| $\lambda \sim \ell$ | Complex dynamics |
| $\lambda \gg \ell$ | Local dynamics; population tracks habitat |

## Source-Sink Dynamics

Heterogeneous landscapes create:
- **Sources**: Good habitat exporting individuals
- **Sinks**: Poor habitat requiring immigration

Overall population persistence depends on balance.

## Applications

1. **Conservation**: Reserve design and corridors
2. **Agriculture**: Crop pest dynamics in patchy fields
3. **Climate change**: Shifting habitat suitability
4. **Urban ecology**: Green space connectivity
5. **Disease ecology**: Heterogeneous transmission

## Critical Spatial Scales

**Critical patch size**: Minimum favorable area for persistence
**Critical connectivity**: Required corridors between patches
**Critical heterogeneity**: Amplitude beyond which extinction occurs

## Analytical Insights

For rapidly varying environment ($f$ large):
- Effective medium approximation
- Homogenization theory applies
- Population sees averaged parameters

For slowly varying ($f$ small):
- Local equilibrium in each region
- Gradients at patch boundaries

## References

- Shigesada, N. & Kawasaki, K. (1997). *Biological Invasions: Theory and Practice*
- Cantrell, R.S. & Cosner, C. (2003). *Spatial Ecology via Reaction-Diffusion Equations*
- Ovaskainen, O. & Meerson, B. (2010). *Stochastic models of population extinction*

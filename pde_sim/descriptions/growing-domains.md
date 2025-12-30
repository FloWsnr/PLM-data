# Patterns on Growing Domains

## Mathematical Formulation

Reaction-diffusion on effectively growing domain via dilution:

$$\frac{\partial u}{\partial t} = D_u \nabla^2 u + a - u + u^2 v - \rho u$$
$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + b - u^2 v - \rho v$$

where:
- $u, v$ are chemical concentrations
- $D_u, D_v$ are diffusion coefficients
- $a, b$ are production rates
- $\rho$ is the growth/dilution rate

## Physical Background

Domain growth creates **dilution**: as the domain expands, concentrations decrease if production doesn't compensate.

The $-\rho u$ and $-\rho v$ terms model this dilution effect.

**Actual domain growth** would involve:
- Moving boundaries
- Advection terms from expansion
- Complex coordinate transformations

The dilution approximation captures key effects simply.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Activator diffusion | $D_u$ | Small (slow) | 0.01 - 1 |
| Inhibitor diffusion | $D_v$ | Large (fast) | 0.1 - 50 |
| Production a | $a$ | Activator source | 0 - 1 |
| Production b | $b$ | Inhibitor source | 0 - 2 |
| Growth rate | $\rho$ | Dilution rate | 0 - 0.5 |

## Growth Effects on Patterns

| Growth Rate | Effect |
|-------------|--------|
| $\rho = 0$ | Standard Turing patterns |
| Small $\rho$ | Patterns adapt, wavelength may change |
| Moderate $\rho$ | New patterns emerge during growth |
| Large $\rho$ | Patterns suppressed, dilution dominates |

## Biological Relevance

Real organisms grow, so patterns form on **expanding domains**:

1. **Embryonic development**: Tissue grows during patterning
2. **Limb formation**: Digits form as limb bud grows
3. **Skin patterns**: Scale/fur patterns on growing skin
4. **Leaf venation**: Veins form as leaf expands

## Pattern Sequence

On growing domains, patterns may undergo **mode transitions**:
- Early: Uniform or few spots
- Intermediate: More spots appear
- Late: Full Turing pattern

New spots appear as domain grows large enough to support more wavelengths.

## Peak Insertion/Splitting

Growth can trigger:
- **Peak splitting**: One spot → two spots
- **Peak insertion**: New spot between existing
- **Mode doubling**: Pattern wavelength halves

## Cramér's Model

For truly growing domain with velocity $\mathbf{V}$:
$$\frac{\partial u}{\partial t} + \nabla \cdot (u\mathbf{V}) = D\nabla^2 u + f(u,v)$$

Uniform growth: $\mathbf{V} = \rho \mathbf{r}$ (radial expansion)

## Applications

1. **Developmental biology**: Morphogen patterns
2. **Cancer modeling**: Tumor growth patterns
3. **Bacterial colonies**: Expanding population patterns
4. **Wound healing**: Pattern formation in regeneration
5. **Plant development**: Phyllotaxis, leaf patterns

## Theoretical Insight

Growth can:
- Trigger pattern onset (domain becomes large enough)
- Select wavelength (by growth rate matching)
- Create historical dependence (pattern remembers past sizes)

## Numerical Considerations

- Fixed domain with dilution is simplest
- Time-dependent patterns require long integration
- Track pattern changes during evolution
- Consider growth rate vs. reaction rate timescales

## References

- Crampin, E.J. et al. (1999). *Reaction and diffusion on growing domains*
- Painter, K.J. et al. (1999). *Stripe formation in juvenile Pomacanthus*
- Plaza, R.G. et al. (2004). *The effect of growth and curvature on pattern formation*

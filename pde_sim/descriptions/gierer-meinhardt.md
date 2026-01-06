# Gierer-Meinhardt Model

## Mathematical Formulation

The Gierer-Meinhardt activator-inhibitor system (visualpde.com formulation):

$$\frac{\partial u}{\partial t} = D_u \nabla^2 u + a + \frac{u^2}{v} - bu$$
$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + u^2 - cv$$

where:
- $u$ is the activator concentration
- $v$ is the inhibitor concentration
- $D_u, D_v$ are diffusion coefficients ($D_v \gg D_u$)
- $a$ is basal activator production
- $b$ is activator decay rate
- $c$ is inhibitor decay rate

## Physical Background

The Gierer-Meinhardt model is a classic model for biological pattern formation. Key mechanisms:

1. **Short-range activation**: Activator enhances its own production locally
2. **Long-range inhibition**: Inhibitor produced by activator, diffuses faster
3. **Local self-enhancement**: $u^2/v$ term creates positive feedback
4. **Lateral inhibition**: Fast-diffusing inhibitor suppresses distant activation

This mechanism was proposed to explain morphogenesis and regeneration in Hydra.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Activator diffusion | $D_u$ | Short-range (slow) | 0.1 - 10 |
| Inhibitor diffusion | $D_v$ | Long-range (fast) | 1 - 200 |
| Basal production | $a$ | Background activator source | 0 - 1 |
| Activator decay | $b$ | Degradation rate | 0.1 - 5 |
| Inhibitor decay | $c$ | Degradation rate | 0.1 - 5 |

## Turing Instability Conditions

For pattern formation:
1. **Diffusion ratio**: $D_v/D_u \gg 1$ (typically 10-1000)
2. **Decay balance**: Appropriate $\mu_u/\mu_v$ ratio
3. **Wavenumber selection**: Specific pattern wavelength emerges

## Pattern Types

- **Spots**: Isolated peaks of activator
- **Stripes**: Extended activation regions
- **Peaks on boundaries**: Activation at domain edges
- **Branching structures**: Tip-splitting patterns

## Biological Applications

1. **Hydra regeneration**: Head and foot formation
2. **Embryonic patterning**: Body axis formation
3. **Leaf venation**: Vascular pattern formation
4. **Hair follicle spacing**: Regular arrangements
5. **Limb development**: Digit patterning

## Special Cases

**No basal production** ($\rho_u = 0$):
- Activator can go extinct
- Patterns require initial activation

**With basal production** ($\rho_u > 0$):
- Maintains minimum activator level
- More robust pattern formation

## Numerical Considerations

- Division by $v$ can cause issues if $v \to 0$
- Add small regularization: $u^2/(v + \epsilon)$
- Explicit schemes work for moderate diffusion ratios

## References

- Gierer, A. & Meinhardt, H. (1972). *A Theory of Biological Pattern Formation*
- Meinhardt, H. (1982). *Models of Biological Pattern Formation*
- Meinhardt, H. (2008). *Models of Biological Pattern Formation: From Elementary Steps to the Organization of Embryonic Axes*

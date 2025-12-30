# Gray-Scott Reaction-Diffusion System

## Mathematical Formulation

The Gray-Scott model describes autocatalytic reactions:

$$\frac{\partial u}{\partial t} = D_u \nabla^2 u - uv^2 + F(1 - u)$$
$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + uv^2 - (F + k)v$$

where:
- $u$ is the concentration of "feed" chemical
- $v$ is the concentration of "catalyst" chemical
- $D_u, D_v$ are diffusion coefficients
- $F$ is the feed rate (inflow rate)
- $k$ is the kill rate (outflow rate)

## Physical Background

The Gray-Scott model represents a continuously stirred tank reactor (CSTR):

1. **Feed**: Fresh $u$ enters at concentration 1, rate $F$
2. **Reaction**: $u + 2v \to 3v$ (autocatalytic)
3. **Removal**: Both species leave at rate $F$ (plus extra $k$ for $v$)

The autocatalytic term $uv^2$ creates positive feedback driving pattern formation.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Feed rate | $F$ | Replenishment rate | 0 - 0.1 |
| Kill rate | $k$ | Removal rate of v | 0 - 0.1 |
| Diffusion u | $D_u$ | Feed diffusion | 0.01 - 1 |
| Diffusion v | $D_v$ | Catalyst diffusion | 0.01 - 1 |

## Pearson's Classification

Different $(F, k)$ values produce remarkably diverse patterns:

| Pattern | Description |
|---------|-------------|
| α (alpha) | Traveling waves |
| β (beta) | Hexagonal spots |
| γ (gamma) | Stripes/labyrinths |
| δ (delta) | Spots that replicate |
| ε (epsilon) | Spots and worms |
| ζ (zeta) | Chaos |
| η (eta) | Solitons |
| θ (theta) | Pulsating spots |
| ι (iota) | Negative spots |
| κ (kappa) | Self-replicating spots |
| λ (lambda) | Mitosis-like behavior |
| μ (mu) | Mixed patterns |

## Classic Parameter Sets

| Name | F | k | Pattern |
|------|---|---|---------|
| Solitons | 0.030 | 0.062 | Moving spots |
| Mitosis | 0.028 | 0.062 | Dividing spots |
| Coral | 0.055 | 0.062 | Branching growth |
| Spirals | 0.014 | 0.054 | Spiral waves |
| Zebra | 0.035 | 0.065 | Stripes |
| Fingerprint | 0.037 | 0.060 | Labyrinths |

## Initialization

Typical initial condition:
- $u = 1$ everywhere (saturated with feed)
- $v = 0$ everywhere except perturbation region
- Perturbation seeds pattern growth

## Applications

1. **Chemical engineering**: CSTR reactor patterns
2. **Computational art**: Generative graphics
3. **Biology**: Morphogenesis analogy
4. **Materials science**: Self-organized textures
5. **Nonlinear dynamics**: Paradigm for complexity

## Numerical Considerations

- Standard explicit schemes work well
- Periodic boundaries typical
- Resolution affects pattern wavelength
- Long integration times needed for steady patterns

## References

- Gray, P. & Scott, S.K. (1984). *Autocatalytic reactions in the isothermal, continuous stirred tank reactor*
- Pearson, J.E. (1993). *Complex Patterns in a Simple System*, Science 261
- Munafo, R. (2014). *Gray-Scott pattern gallery* (online resource)

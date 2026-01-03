# SIR Epidemiological Model

## Mathematical Formulation

The spatially coupled SIR model describes disease spread:

$$\frac{\partial s}{\partial t} = D \nabla^2 s - \beta i s$$
$$\frac{\partial i}{\partial t} = D \nabla^2 i + \beta i s - \gamma i$$
$$\frac{\partial r}{\partial t} = D \nabla^2 r + \gamma i$$

where:
- $s$ is the density of susceptible individuals
- $i$ is the density of infected individuals
- $r$ is the density of recovered (immune) individuals
- $D$ is the diffusivity (spatial mobility)
- $\beta$ is the transmission rate
- $\gamma$ is the recovery rate

## Physical Background

The SIR model divides a population into three compartments:

1. **Susceptible (S)**: Can catch the disease
2. **Infected (I)**: Currently infected and infectious
3. **Recovered (R)**: Immune (recovered or deceased)

Key processes:
- **Infection**: $S + I \to 2I$ at rate $\beta$
- **Recovery**: $I \to R$ at rate $\gamma$
- **Diffusion**: Spatial movement at rate $D$

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Transmission | $\beta$ | Infection rate | 0.1 - 5 |
| Recovery | $\gamma$ | Recovery rate | 0.01 - 2 |
| Diffusivity | $D$ | Spatial mobility | 0.001 - 0.3 |

## Key Quantities

| Quantity | Formula | Meaning |
|----------|---------|---------|
| Basic reproduction number | $R_0 = \beta/\gamma$ | Average secondary infections |
| Herd immunity threshold | $1 - 1/R_0$ | Fraction needed for protection |
| Wave speed | $c \approx 2\sqrt{D\beta}$ | Epidemic front velocity |

## Conservation Law

The total population is conserved:
$$s + i + r = \text{constant}$$

This conservation arises from the SIR model's closed population assumption.

## Epidemic Dynamics

| $R_0$ Value | Behavior |
|-------------|----------|
| $R_0 < 1$ | Epidemic dies out |
| $R_0 > 1$ | Epidemic spreads |
| $R_0 \gg 1$ | Rapid, widespread outbreak |

## Spatial Patterns

With diffusion, the model exhibits:
- **Traveling waves**: Epidemic fronts spreading outward
- **Endemic equilibria**: Stable disease presence
- **Spatial heterogeneity**: Localized outbreaks

## Initialization

Common initial conditions:
- **Localized infection**: Small infected region in susceptible population
- **Random seeding**: Multiple infection foci
- **Uniform**: Homogeneous infected fraction

## Applications

1. **Public health**: Epidemic forecasting
2. **Veterinary science**: Animal disease spread
3. **Plant pathology**: Crop disease dynamics
4. **Computer viruses**: Malware propagation
5. **Information spread**: Rumor/meme dynamics

## Model Extensions

| Extension | Additional Compartment |
|-----------|----------------------|
| SEIR | Exposed (latent period) |
| SIRS | Waning immunity |
| SIS | No immunity |
| SIRD | Death compartment |

## Numerical Considerations

- Non-negative initial conditions essential
- Explicit Euler often sufficient
- Mass conservation provides numerical check
- Periodic or no-flux boundaries typical

## References

- Kermack, W.O. & McKendrick, A.G. (1927). *A Contribution to the Mathematical Theory of Epidemics*, Proc. R. Soc. London A
- Murray, J.D. (2002). *Mathematical Biology*, Chapter 10-11, Springer
- Keeling, M.J. & Rohani, P. (2008). *Modeling Infectious Diseases*, Princeton University Press

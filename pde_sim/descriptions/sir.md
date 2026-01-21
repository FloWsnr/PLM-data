# SIR Epidemic Model

A reaction-diffusion extension of the classic Susceptible-Infected-Recovered (SIR) compartmental model, capturing spatial epidemic wave propagation.

## Description

The SIR model is the foundational mathematical framework for understanding infectious disease dynamics. This spatial extension adds diffusion to model geographic spread:

- **Susceptible (S)**: Individuals who can contract the disease
- **Infected (I)**: Individuals currently infectious and capable of spreading disease
- **Recovered (R)**: Individuals who have recovered and are now immune

The model demonstrates several key epidemiological phenomena:

- **Traveling epidemic waves**: Infection fronts propagate through susceptible populations at characteristic speeds
- **Herd immunity threshold**: Wave propagation stops when the susceptible fraction drops below the critical threshold 1/R0
- **Wave speed**: Proportional to sqrt(D_I * beta * S0) where S0 is the initial susceptible fraction
- **Final size relation**: Determines the total fraction of population ultimately infected

The basic reproduction number R0 = beta/gamma represents the average number of secondary infections caused by one infected individual in a fully susceptible population:
- R0 > 1: Epidemic occurs, infection spreads
- R0 < 1: Disease dies out exponentially
- R0 = 1: Critical threshold, endemic equilibrium

Key insight: The spatial model exhibits **wave solutions** that travel at a minimum speed determined by Fisher-KPP theory:
$$c_{min} = 2\sqrt{D_I \cdot \beta \cdot S_0 - D_I \cdot \gamma}$$

When D_S = D_I = D_R = D (equal diffusion), the total population S + I + R remains constant everywhere, preserving the conservation law of the non-spatial model.

## Equations

$$\frac{\partial S}{\partial t} = D_S \nabla^2 S - \beta S I$$

$$\frac{\partial I}{\partial t} = D_I \nabla^2 I + \beta S I - \gamma I$$

$$\frac{\partial R}{\partial t} = D_R \nabla^2 R + \gamma I$$

Where:
- $S$ = susceptible population density
- $I$ = infected population density
- $R$ = recovered (immune) population density
- $\beta$ = transmission/infection rate (contact rate times transmission probability)
- $\gamma$ = recovery rate (1/$\gamma$ = mean infectious period)
- $D_S, D_I, D_R$ = diffusion coefficients for each compartment

## Default Config

```yaml
solver: euler
dt: 0.001
resolution: 128
domain_size: 100.0

bc:
  x-: neumann:0
  x+: neumann:0
  y-: neumann:0
  y+: neumann:0

parameters:
  beta: 0.5       # Transmission rate
  gamma: 0.1      # Recovery rate (R0 = 5)
  D_S: 0.1        # Susceptible diffusion
  D_I: 0.1        # Infected diffusion
  D_R: 0.1        # Recovered diffusion

init:
  type: wave
  params:
    S0: 0.99       # Initial susceptible fraction
    I0: 0.01       # Peak infected in seed region
    seed_radius: 5.0
```

## Parameter Variants

### Default (Epidemic Wave)
Demonstrates classic traveling epidemic wave with R0 = 5:
- High transmission rate (beta = 0.5)
- Moderate recovery (gamma = 0.1)
- Clear wave front propagation

### Mild Epidemic (R0 ~ 2)
Lower transmission rate (beta = 0.2, gamma = 0.1):
- Slower wave propagation
- Lower final attack rate
- More gradual epidemic curve

### High Diffusion
Increased mobility (D_I = 1.0):
- Faster spatial spread
- More diffuse infection front
- Earlier peak at distant locations

### Contained Outbreak (R0 < 1)
Parameters where disease cannot sustain transmission:
- beta = 0.05, gamma = 0.1 gives R0 = 0.5
- Initial seed decays without propagating
- Demonstrates epidemic threshold

### Superspreader (High R0)
Very infectious disease (beta = 1.0, gamma = 0.1, R0 = 10):
- Rapid wave propagation
- Sharp infection front
- High final attack rate

## Applications

1. **COVID-19 modeling**: Spatial spread between regions
2. **Historical epidemics**: Plague, influenza wave propagation
3. **Wildlife disease**: Rabies fronts in fox populations
4. **Plant disease**: Agricultural pathogen spread

## Notes

- The model assumes homogeneous mixing within local neighborhoods
- No vital dynamics (births/deaths) - appropriate for fast epidemics
- Extension: SEIR adds Exposed (latent) compartment between S and I
- Extension: SIS model for diseases without immunity (R returns to S)
- Neumann (no-flux) boundaries model isolated populations
- Periodic boundaries model toroidal/continuous space

## Conservation Law

When D_S = D_I = D_R, integrating over the domain:
$$\frac{d}{dt}\int(S + I + R)\,dA = D\int\nabla^2(S+I+R)\,dA = 0$$
(with no-flux boundaries)

The total population is conserved, matching the well-mixed ODE model.

## References

- Kermack, W. O., & McKendrick, A. G. (1927). A contribution to the mathematical theory of epidemics. Proceedings of the Royal Society A, 115(772), 700-721.
- Murray, J. D. (2002). Mathematical Biology II: Spatial Models and Biomedical Applications. Springer.
- Noble, J. V. (1974). Geographic and temporal development of plagues. Nature, 250(5469), 726-728.
- Brauer, F., & Castillo-Chavez, C. (2012). Mathematical Models in Population Biology and Epidemiology. Springer.

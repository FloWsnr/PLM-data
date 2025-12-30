# Cyclic Competition Model (Rock-Paper-Scissors)

## Mathematical Formulation

The cyclic competition model for three species:

$$\frac{\partial u}{\partial t} = D \nabla^2 u + u(1 - u - v - w) - \alpha uv$$
$$\frac{\partial v}{\partial t} = D \nabla^2 v + v(1 - u - v - w) - \alpha vw$$
$$\frac{\partial w}{\partial t} = D \nabla^2 w + w(1 - u - v - w) - \alpha wu$$

where:
- $u, v, w$ are species densities
- $D$ is the diffusion coefficient
- $\alpha$ is the competition strength
- Competition follows: $u \to v \to w \to u$ (rock-paper-scissors)

## Physical Background

This model implements **intransitive competition**: no single species dominates all others.

- Species $u$ outcompetes $v$
- Species $v$ outcompetes $w$
- Species $w$ outcompetes $u$

This cyclic hierarchy prevents competitive exclusion and maintains biodiversity.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Diffusion | $D$ | Movement rate | 0.001 - 10 |
| Competition | $\alpha$ | Interaction strength | 0.1 - 10 |

## Coexistence Mechanism

Without space (ODE): System exhibits heteroclinic cycles or coexistence point.

With space (PDE): **Spiral waves** emerge that maintain all three species:
- Species chase each other around spiral arms
- Spatial segregation prevents extinction
- Mobility (diffusion) is essential for coexistence

## Spiral Wave Formation

Spontaneous formation of rotating spiral waves:
1. Local fluctuation creates asymmetry
2. Cyclic dominance creates rotation
3. Spirals self-organize and interact
4. Coarsening: small spirals absorbed by large ones

## Applications

1. **Microbial ecology**: Colicin-producing E. coli strains
2. **Coral reef competition**: Space competition between corals
3. **Lizard mating strategies**: Side-blotched lizards
4. **Game theory**: Evolutionary games with cyclic payoffs
5. **Epidemiology**: Competing pathogen strains

## May-Leonard Model

A related model with different dynamics:
- Reproduction separate from competition
- Different mobility for different processes
- Richer dynamics including vortex pinning

## Biodiversity Maintenance

The cyclic competition model demonstrates how:
- Spatial structure promotes coexistence
- Mobility affects diversity
- "Survival of the weakest" can occur (least competitive persists)

## Numerical Considerations

- Three fields must remain non-negative
- Initial conditions affect spiral handedness
- Large domains needed for multiple spirals
- Long-time simulations for coarsening dynamics

## References

- Kerr, B. et al. (2002). *Local dispersal promotes biodiversity in a real-life game of rock-paper-scissors*
- Reichenbach, T., Mobilia, M., & Frey, E. (2007). *Mobility promotes and jeopardizes biodiversity*
- May, R.M. & Leonard, W.J. (1975). *Nonlinear aspects of competition between three species*

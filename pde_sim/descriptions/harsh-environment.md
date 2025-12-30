# Harsh Environment Model (Strong Allee Effect)

## Mathematical Formulation

Population dynamics with strong Allee effect:

$$\frac{\partial u}{\partial t} = D \nabla^2 u + ru(u - \theta)(1 - u)$$

where:
- $u$ is normalized population density (0 to 1)
- $D$ is diffusion coefficient
- $r$ is growth rate
- $\theta$ is the Allee threshold (0 < $\theta$ < 1)

## Physical Background

The **Allee effect** describes positive density dependence at low populations:
- Below threshold $\theta$: Population declines toward extinction
- Above threshold $\theta$: Population grows toward carrying capacity

This creates **bistability**: two stable states (extinction and carrying capacity) with an unstable threshold between them.

## Allee Effect Origins

Mechanisms causing Allee effects:
1. **Mate finding**: Difficulty finding partners at low density
2. **Cooperative defense**: Vulnerability to predators
3. **Cooperative hunting**: Insufficient hunting success
4. **Inbreeding**: Genetic depression in small populations
5. **Environmental modification**: Loss of habitat improvement

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Diffusion | $D$ | Movement rate | 0.001 - 10 |
| Growth rate | $r$ | Intrinsic reproduction | 0.1 - 10 |
| Threshold | $\theta$ | Allee threshold | 0 - 0.5 |

## Traveling Wave Dynamics

Unlike Fisher-KPP, this equation supports **pushed waves**:

- **Pulled wave** (Fisher-KPP): Speed determined by leading edge
- **Pushed wave** (Allee): Speed determined by bulk dynamics

Wave speed depends on threshold:
- Larger $\theta$ → slower or retreating waves
- Smaller $\theta$ → faster advancing waves

## Critical Patch Size

For bounded domains, there exists a **critical patch size**:
- Below critical: Population always goes extinct
- Above critical: Population can persist

This has important conservation implications.

## Bistability in Space

Initial condition matters greatly:
- Small patch below threshold: Extinction
- Large patch above threshold: Expansion to filling domain
- Intermediate: Depends on exact shape and size

## Applications

1. **Conservation biology**: Minimum viable populations
2. **Invasive species**: Establishment thresholds
3. **Fisheries**: Stock collapse and recovery
4. **Reintroduction**: Required population sizes
5. **Habitat fragmentation**: Patch viability

## Ecological Implications

- **Extinction debt**: Populations doomed but not yet extinct
- **Hysteresis**: Different thresholds for collapse vs. recovery
- **Range limits**: Allee effects determine boundaries
- **Rescue effects**: Immigration preventing local extinction

## References

- Lewis, M.A. & Kareiva, P. (1993). *Allee dynamics and the spread of invading organisms*
- Courchamp, F. et al. (2008). *Allee Effects in Ecology and Conservation*
- Taylor, C.M. & Hastings, A. (2005). *Allee effects in biological invasions*

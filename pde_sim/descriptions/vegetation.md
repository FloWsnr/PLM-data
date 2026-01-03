# Klausmeier Vegetation Model

## Mathematical Formulation

The Klausmeier model for vegetation-water dynamics on slopes:

$$\frac{\partial w}{\partial t} = a - w - wn^2 + v\frac{\partial w}{\partial x}$$
$$\frac{\partial n}{\partial t} = D_n \nabla^2 n + wn^2 - mn$$

where:
- $w$ is water density
- $n$ is vegetation (plant) density
- $a$ is rainfall rate
- $v$ is water flow velocity (downslope)
- $m$ is plant mortality rate
- $D_n$ is plant dispersal coefficient

## Physical Background

The Klausmeier model explains striking **banded vegetation patterns** observed in semi-arid regions:

1. **Water flows downhill**: Advection term $v\partial w/\partial x$
2. **Plants trap water**: Water consumption $-wn^2$
3. **Plants grow with water**: Growth term $+wn^2$
4. **Plants die**: Mortality $-mn$
5. **Plants spread**: Diffusion $D_n\nabla^2 n$

The interplay creates bands perpendicular to slope direction.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Rainfall | $a$ | Water input rate | 0.1 - 5 |
| Flow velocity | $v$ | Downslope water speed | 0 - 2 |
| Mortality | $m$ | Plant death rate | 0.01 - 2 |
| Dispersal | $D_n$ | Plant seed spread | 0.001 - 0.1 |

## Pattern Types

Depending on parameters (especially rainfall $a$):
1. **Bare soil**: Low rainfall, no vegetation
2. **Uniform vegetation**: High rainfall
3. **Bands/stripes**: Intermediate rainfall
4. **Spots**: Near desertification threshold
5. **Labyrinths**: Transition patterns

## Tiger Bush Phenomenon

The model explains "Tiger Bush" or "brousse tigrée":
- Alternating bands of vegetation and bare soil
- Bands align perpendicular to slopes
- Migration upslope over time
- Observed in Africa, Australia, Americas

## Ecological Significance

The patterns indicate:
- **Ecosystem stress**: Patterns emerge under water limitation
- **Early warning**: Pattern changes predict desertification
- **Resilience**: Patterns are more stable than uniform vegetation

## Rainfall-Pattern Relationship

| Rainfall | Pattern |
|----------|---------|
| Very low | Desert (bare soil) |
| Low | Spots |
| Medium-low | Labyrinths |
| Medium | Bands/stripes |
| High | Uniform vegetation |

## Applications

1. **Desertification monitoring**: Tracking ecosystem health
2. **Land management**: Predicting vegetation changes
3. **Climate change**: Assessing aridity impacts
4. **Conservation**: Identifying vulnerable areas

## References

- Klausmeier, C.A. (1999). *Regular and Irregular Patterns in Semiarid Vegetation*
- Rietkerk, M. & van de Koppel, J. (2008). *Regular pattern formation in real ecosystems*
- Meron, E. (2012). *Pattern-Formation Approach to Dryland Vegetation*

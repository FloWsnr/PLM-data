# Population Dynamics on Terrain

## Mathematical Formulation

Population dynamics with terrain-influenced movement:

$$\frac{\partial u}{\partial t} = D \nabla^2 u - \alpha \nabla u \cdot \nabla h + ru(1 - u)$$

Simplified with constant slope:
$$\frac{\partial u}{\partial t} = D \nabla^2 u - \alpha\left(h_x \frac{\partial u}{\partial x} + h_y \frac{\partial u}{\partial y}\right) + ru(1 - u)$$

where:
- $u$ is population density
- $D$ is diffusion coefficient
- $\alpha$ is terrain influence strength
- $h(x,y)$ is elevation (or terrain effect)
- $h_x, h_y$ are slope components
- $r$ is growth rate

## Physical Background

Organisms respond to terrain through:
1. **Gravity effects**: Easier movement downhill
2. **Energy costs**: Slope affects travel efficiency
3. **Habitat preference**: Elevation correlates with conditions
4. **Water availability**: Flows downhill, affects resources

The term $-\alpha \nabla u \cdot \nabla h$ creates advection **down** gradients of $h$.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Diffusion | $D$ | Random movement | 0.01 - 0.5 |
| Terrain influence | $\alpha$ | Slope response | 0.0 - 2.0 |
| Growth rate | $r$ | Reproduction | 0.1 - 2.0 |
| Slope x | $h_x$ | East-west slope | -1 to 1 |
| Slope y | $h_y$ | North-south slope | -1 to 1 |

## Movement Interpretation

**Positive $\alpha$**: Population moves toward **lower** elevation
- Models gravity-driven drift
- Water-seeking behavior
- Energy-minimizing movement

**Negative $\alpha$**: Population moves toward **higher** elevation
- Models altitude preference
- Temperature-seeking behavior
- Escape from flooding

## Equilibrium Distribution

At steady state, population concentrates in:
- **Valleys** if $\alpha > 0$
- **Peaks** if $\alpha < 0$
- Balance between drift, diffusion, and growth

## Applications

1. **Wildlife corridors**: Terrain-guided movement paths
2. **Mountain ecology**: Elevation gradient patterns
3. **Watershed dynamics**: Species following water
4. **Climate migration**: Upslope movement with warming
5. **Urban ecology**: Populations in hilly cities

## Terrain Types

| Terrain | Effect |
|---------|--------|
| Valley | Population accumulation |
| Ridge | Population depletion |
| Saddle | Complex flow patterns |
| Bowl | Concentration center |
| Slope | Directional drift |

## Extensions

More realistic models include:
- **Variable terrain**: Full topographic maps
- **Barriers**: Cliffs, water bodies
- **Corridors**: Mountain passes
- **Aspect**: Sun exposure effects

## References

- Cantrell, R.S. & Cosner, C. (2003). *Spatial Ecology via Reaction-Diffusion Equations*
- Potapov, A. & Lewis, M.A. (2004). *Climate and competition: effects of habitat gradient*

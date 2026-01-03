# Bacteria Chemotaxis in Flow

## Mathematical Formulation

Bacteria with chemotaxis in flowing medium:

$$\frac{\partial b}{\partial t} = D_b \nabla^2 b - \chi \nabla \cdot (b \nabla c) - v\frac{\partial b}{\partial x} + rb$$
$$\frac{\partial c}{\partial t} = D_c \nabla^2 c - kc + sb$$

where:
- $b$ is bacteria density
- $c$ is chemoattractant concentration
- $D_b, D_c$ are diffusion coefficients
- $\chi$ is chemotactic sensitivity
- $v$ is flow velocity
- $r$ is bacteria growth rate
- $k$ is chemical decay rate
- $s$ is chemical production rate

## Physical Background

This model combines:
1. **Chemotaxis**: Bacteria swim toward chemical signals
2. **Advection**: Background flow carries bacteria
3. **Growth**: Bacterial reproduction
4. **Signaling**: Bacteria produce attractant

The interplay creates complex spatial-temporal dynamics.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Bacteria diffusion | $D_b$ | Random swimming | 0.01 - 1 |
| Chemical diffusion | $D_c$ | Signal spread | 0.01 - 0.5 |
| Chemotaxis | $\chi$ | Attraction strength | 0.0 - 2.0 |
| Flow velocity | $v$ | Background flow | 0.0 - 2.0 |
| Growth rate | $r$ | Reproduction | 0 - 2 |
| Decay rate | $k$ | Signal degradation | 0.01 - 2 |
| Production | $s$ | Signal generation | 0 - 2 |

## Dynamics Regimes

**No flow** ($v = 0$):
- Chemotactic aggregation (Keller-Segel dynamics)
- Possible blow-up or pattern formation

**With flow** ($v > 0$):
- Advected aggregations
- Traveling bands and plumes
- Upstream swimming against flow
- Biofilm streamer formation

## Traveling Band Solutions

Under certain conditions, bacteria form **traveling bands**:
- Self-organized waves of bacteria
- Consume and produce attractant
- Move at characteristic speed
- Can move upstream against flow

## Applications

1. **Biofilm formation**: Initial colonization patterns
2. **Bacterial infections**: Pathogen migration
3. **Environmental remediation**: Bioremediation in groundwater
4. **Microfluidics**: Lab-on-chip devices
5. **Marine biology**: Bacterial plumes near nutrients

## Experimental Context

Models phenomena observed in:
- E. coli swimming patterns
- Pseudomonas biofilm streamers
- Salmonella swarming
- Marine bacteria nutrient tracking

## Flow Effects

| Flow Strength | Bacterial Response |
|---------------|-------------------|
| None | Aggregation/blow-up |
| Weak | Elongated colonies |
| Moderate | Traveling bands |
| Strong | Washout or upstream bands |

## References

- Keller, E.F. & Segel, L.A. (1971). *Traveling bands of chemotactic bacteria*
- Tuval, I. et al. (2005). *Bacterial swimming and oxygen transport*
- Rusconi, R. et al. (2014). *Bacterial transport suppressed by fluid shear*

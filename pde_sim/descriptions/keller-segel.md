# Keller-Segel Chemotaxis Model

## Mathematical Formulation

The Keller-Segel system describes cell movement toward chemical gradients:

$$\frac{\partial u}{\partial t} = D_u \nabla^2 u - \chi \nabla \cdot (u \nabla c)$$
$$\frac{\partial c}{\partial t} = D_c \nabla^2 c + \alpha u - \beta c$$

where:
- $u$ is the cell (bacteria/amoeba) density
- $c$ is the chemoattractant concentration
- $D_u, D_c$ are diffusion coefficients
- $\chi$ is the chemotactic sensitivity
- $\alpha$ is chemoattractant production rate
- $\beta$ is chemoattractant decay rate

## Physical Background

Chemotaxis is the directed movement of cells in response to chemical gradients. The Keller-Segel model captures:

1. **Random motility**: Diffusion of cells ($D_u \nabla^2 u$)
2. **Chemotaxis**: Biased movement up gradients ($-\chi \nabla \cdot (u \nabla c)$)
3. **Signal production**: Cells produce attractant ($\alpha u$)
4. **Signal decay**: Attractant degrades ($-\beta c$)

## Blow-up Phenomenon

A remarkable feature of the Keller-Segel system is **finite-time blow-up**:
- In 2D: Blow-up occurs if initial mass exceeds critical threshold: $\int u dx > 8\pi D_u/\chi$
- Cell density becomes singular (delta function)
- Models extreme aggregation

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Cell diffusion | $D_u$ | Random motility | 0.01 - 10 |
| Chemical diffusion | $D_c$ | Signal spread | 0.01 - 10 |
| Chemotactic sensitivity | $\chi$ | Attraction strength | 0 - 10 |
| Production rate | $\alpha$ | Signal generation | 0 - 10 |
| Decay rate | $\beta$ | Signal degradation | 0.01 - 10 |

## Dynamics Regimes

1. **Diffusion-dominated** ($\chi$ small): Cells spread uniformly
2. **Chemotaxis-dominated** ($\chi$ large): Aggregation, possible blow-up
3. **Balanced**: Pattern formation, traveling bands

## Applications

1. **Bacterial swarming**: E. coli colony patterns
2. **Slime mold aggregation**: Dictyostelium discoideum
3. **Wound healing**: Neutrophil migration
4. **Tumor angiogenesis**: Endothelial cell movement
5. **Immune response**: Lymphocyte homing

## Model Variants

- **Volume-filling**: Prevents blow-up with crowding effects
- **Signal-dependent sensitivity**: $\chi = \chi(c)$
- **Multiple populations**: Competing or cooperating species
- **Nonlocal sensing**: Gradient detection over finite range

## Numerical Considerations

- Near blow-up, extreme resolution needed
- Positivity preservation important
- Consider volume-filling modifications for robustness
- Adaptive mesh refinement beneficial

## References

- Keller, E.F. & Segel, L.A. (1970). *Initiation of Slime Mold Aggregation*
- Keller, E.F. & Segel, L.A. (1971). *Model for Chemotaxis*
- Horstmann, D. (2003). *From 1970 until present: the Keller-Segel model in chemotaxis*

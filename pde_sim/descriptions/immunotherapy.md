# Tumor-Immune Interaction Model

## Mathematical Formulation

A simplified tumor-immune dynamics model:

$$\frac{\partial T}{\partial t} = D_T \nabla^2 T + r_T T\left(1 - \frac{T}{K}\right) - kTI$$
$$\frac{\partial I}{\partial t} = D_I \nabla^2 I + s + \frac{r_I TI}{a + T} - d_I I$$

where:
- $T$ is tumor cell density
- $I$ is immune cell density
- $D_T, D_I$ are diffusion coefficients
- $r_T$ is tumor growth rate, $K$ is carrying capacity
- $k$ is kill rate of tumor by immune cells
- $s$ is immune cell source (therapy)
- $r_I$ is immune recruitment rate
- $a$ is half-saturation constant
- $d_I$ is immune cell death rate

## Physical Background

This model captures key tumor-immune interactions:

1. **Tumor growth**: Logistic growth to carrying capacity
2. **Immune killing**: Tumor death proportional to T×I
3. **Immune recruitment**: Attracted by tumor (Michaelis-Menten)
4. **Immune death**: Natural turnover
5. **Therapy**: External immune cell source ($s$)

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Tumor diffusion | $D_T$ | Tumor spread rate | 0.001 - 1 |
| Immune diffusion | $D_I$ | Immune motility | 0.001 - 1 |
| Tumor growth | $r_T$ | Proliferation rate | 0.1 - 2.0 |
| Carrying capacity | $K$ | Maximum tumor size | 0.1 - 10 |
| Kill rate | $k$ | Immune effectiveness | 0.1 - 2.0 |
| Immune source | $s$ | Therapy strength | 0 - 2 |
| Recruitment | $r_I$ | Immune attraction | 0.0 - 2.0 |
| Half-saturation | $a$ | Recruitment threshold | 0.01 - 1 |
| Immune death | $d_I$ | Cell turnover rate | 0.01 - 2 |

## Possible Outcomes

1. **Tumor elimination**: Strong immune response clears tumor
2. **Tumor escape**: Tumor grows despite immune response
3. **Dormancy**: Tumor and immune reach equilibrium
4. **Oscillations**: Periodic tumor-immune dynamics
5. **Spatial patterns**: Heterogeneous tumor-immune distribution

## Therapeutic Implications

The source term $s$ models immunotherapy:
- **CAR-T therapy**: Large immune cell injection
- **Checkpoint inhibitors**: Reduced immune death $d_I$
- **Cytokine therapy**: Enhanced recruitment $r_I$

## Spatial Dynamics

Without space (ODE): Classic Kuznetsov model behavior
With space (PDE): Additional phenomena:
- **Invasion fronts**: Tumor spreading into tissue
- **Immune barriers**: Immune cells stopping tumor advance
- **Patchy tumors**: Heterogeneous spatial distribution

## Applications

1. **Cancer immunotherapy**: Treatment optimization
2. **Tumor growth modeling**: Predicting progression
3. **Treatment protocols**: Timing of therapy
4. **Resistance mechanisms**: Understanding escape

## References

- Kuznetsov, V.A. et al. (1994). *Nonlinear dynamics of immunogenic tumors*
- de Pillis, L.G. et al. (2005). *A mathematical tumor-immune system dynamics model*
- Eftimie, R. et al. (2011). *Interactions between the immune system and cancer*

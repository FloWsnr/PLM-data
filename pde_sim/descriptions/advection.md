# Advection-Diffusion Equation

## Mathematical Formulation

The advection-diffusion equation describes transport by both flow and diffusion:

$$\frac{\partial u}{\partial t} = D \nabla^2 u - \mathbf{v} \cdot \nabla u$$

Expanded in 2D:

$$\frac{\partial u}{\partial t} = D \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) - v_x \frac{\partial u}{\partial x} - v_y \frac{\partial u}{\partial y}$$

where:
- $u$ is the transported quantity (concentration, temperature, etc.)
- $D$ is the diffusion coefficient
- $\mathbf{v} = (v_x, v_y)$ is the advection velocity

## Physical Background

This equation combines two transport mechanisms:

1. **Advection**: Bulk transport by fluid flow (hyperbolic character)
2. **Diffusion**: Random molecular motion (parabolic character)

The balance is characterized by the Péclet number: $\text{Pe} = \frac{|\mathbf{v}| L}{D}$
- $\text{Pe} \ll 1$: Diffusion-dominated
- $\text{Pe} \gg 1$: Advection-dominated

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Diffusion coefficient | $D$ | Molecular diffusion rate | 0 - 1 |
| Velocity x | $v_x$ | Advection speed in x | -10 to +10 |
| Velocity y | $v_y$ | Advection speed in y | -10 to +10 |

## Behavior Regimes

### Pure Advection ($D = 0$)
- Solution translates without changing shape
- Discontinuities persist indefinitely
- Numerically challenging (spurious oscillations)

### Advection-Dominated ($\text{Pe} \gg 1$)
- Features advect downstream
- Sharp fronts develop
- Diffusion acts as regularization

### Diffusion-Dominated ($\text{Pe} \ll 1$)
- Features spread and smooth
- Advection causes slow drift

## Applications

1. **Pollutant transport**: Contaminants in rivers, atmosphere
2. **Heat transfer**: Convective cooling/heating
3. **Chemical engineering**: Reactor transport
4. **Oceanography**: Tracer transport in currents
5. **Blood flow**: Drug delivery in vessels

## Numerical Considerations

- High Péclet number requires special schemes (upwinding, SUPG)
- Pure advection benefits from conservative schemes
- Artificial diffusion may be added for stability

## References

- Ferziger, J.H. & Perić, M. (2002). *Computational Methods for Fluid Dynamics*

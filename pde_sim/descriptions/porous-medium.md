# Porous Medium Equation

A nonlinear degenerate diffusion equation modeling gas flow through porous media, groundwater infiltration, and population spreading with density-dependent diffusivity.

## Description

The Porous Medium Equation (PME) is one of the simplest and most studied examples of nonlinear degenerate parabolic equations. It generalizes the linear heat equation by making the diffusion coefficient depend on the solution itself: diffusion is strong where the density is high and vanishes where the density is zero.

Key physical phenomena captured by this equation:
- **Finite speed of propagation**: Unlike the linear heat equation (which has infinite speed of propagation), disturbances in the PME travel at finite speed, producing sharp fronts
- **Compact support**: Solutions that start with compact support remain compactly supported for all time, with the support expanding at a well-defined rate
- **Free boundaries**: The boundary of the support (the "front") moves according to a free boundary condition, making the PME a prototype free boundary problem
- **Self-similar solutions**: The Barenblatt-Pattle solution provides an explicit self-similar solution describing the spreading of a point source

The equation arises in numerous physical contexts:
- **Gas in porous media**: Leibenzon (1930) and Muskat (1937) derived it for gas flow through porous rock, where pressure-dependent density creates nonlinear diffusion
- **Groundwater flow**: The Boussinesq equation for unconfined groundwater reduces to the PME
- **Population dynamics**: Models biological populations with density-dependent dispersal
- **Plasma physics**: Describes the diffusion of a thermal front in a plasma

## Equations

The Porous Medium Equation:

$$\frac{\partial u}{\partial t} = \nabla^2(u^m), \quad m > 1$$

This can be rewritten in divergence form:

$$\frac{\partial u}{\partial t} = \nabla \cdot (m \, u^{m-1} \nabla u)$$

showing that the effective diffusion coefficient is $D_{\text{eff}} = m \, u^{m-1}$, which vanishes at $u = 0$.

Where:
- $u \geq 0$ represents density (gas density, groundwater height, population density)
- $m > 1$ is the nonlinearity exponent

## Default Config

```yaml
solver: euler
dt: 0.001
resolution: [128, 128]
domain_size: [20.0, 20.0]

bc:
  x-: neumann:0
  x+: neumann:0
  y-: neumann:0
  y+: neumann:0

init:
  type: porous-medium-default
  params:
    amplitude: 1.0
    radius: 3.0

parameters:
  m: 2.0
```

## Parameter Variants

### m = 2 (Standard)
The most commonly studied case. Corresponds to gas flow through porous media with polytropic index $\gamma = 2$.

### m = 3 (Stronger nonlinearity)
Produces slower front propagation and steeper density gradients near the free boundary.

### m = 5 (Strong nonlinearity)
Highly nonlinear regime with very sharp fronts and slow spreading. Front speed scales as $t^{1/(m+1)}$.

### Physical Interpretation

| Parameter | Physical Meaning |
|-----------|------------------|
| u | Non-negative density |
| m | Nonlinearity exponent (strength of density-dependent diffusion) |
| m = 1 | Linear heat equation (infinite propagation speed) |
| m > 1 | Porous medium regime (finite propagation speed) |

### Self-Similar Solutions

The Barenblatt-Pattle solution for a point source of mass $M$ in $d$ dimensions:

$$u(x,t) = t^{-\alpha} \left( C - \frac{\alpha(m-1)}{2md} \frac{|x|^2}{t^{2\alpha/d}} \right)_+^{1/(m-1)}$$

where $\alpha = d/(d(m-1)+2)$ and $C$ is determined by mass conservation.

## References

- Vázquez, J.L. (2007). "The Porous Medium Equation: Mathematical Theory" — Oxford University Press
- Barenblatt, G.I. (1952). "On some unsteady motions of a liquid or a gas in a porous medium" — Prikl. Mat. Mekh. 16:67-78
- Muskat, M. (1937). "The Flow of Homogeneous Fluids Through Porous Media" — McGraw-Hill

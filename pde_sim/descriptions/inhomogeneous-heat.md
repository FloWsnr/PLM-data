# Inhomogeneous Heat Equation

## Mathematical Formulation

The inhomogeneous heat equation extends the standard heat equation with a spatially varying source term:

$$\frac{\partial T}{\partial t} = D \nabla^2 T + f(x,y)$$

where the source term $f(x,y)$ is given by:

$$f(x,y) = D \pi^2 \left(\frac{n^2}{L_x^2} + \frac{m^2}{L_y^2}\right) \cos\left(\frac{n\pi x}{L_x}\right) \cos\left(\frac{m\pi y}{L_y}\right)$$

For a square domain with $L = L_x = L_y$:

$$f(x,y) = \frac{D \pi^2 (n^2 + m^2)}{L^2} \cos\left(\frac{n\pi x}{L}\right) \cos\left(\frac{m\pi y}{L}\right)$$

## Physical Background

This equation models heat conduction with a spatially periodic heat source pattern:

- The source term oscillates between positive (heat generation) and negative (heat absorption)
- The integral of $f(x,y)$ over the domain equals zero, ensuring bounded solutions
- Requires homogeneous Neumann (no-flux) boundary conditions for stability

### Key Properties

1. **Bounded Solutions**: Since $\int\int f(x,y) \, dx \, dy = 0$, the average temperature remains constant
2. **Mode Structure**: Parameters $n$ and $m$ control the spatial frequency of the source pattern
3. **Steady State**: The system evolves toward a steady state where diffusion balances the source term

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Diffusion coefficient | $D$ | Rate of heat transport | 0.01 - 0.5 |
| Mode number (x) | $n$ | Spatial frequency in x direction | 1 - 5 |
| Mode number (y) | $m$ | Spatial frequency in y direction | 1 - 5 |

## Boundary Conditions

This PDE requires **homogeneous Neumann (no-flux) boundary conditions**:

$$\frac{\partial T}{\partial n}\bigg|_{\partial\Omega} = 0$$

Using periodic boundary conditions will violate the physical assumptions of this formulation.

## Steady-State Solution

The steady state solution satisfies $D \nabla^2 T = -f(x,y)$, which gives:

$$T_{ss}(x,y) = C + \cos\left(\frac{n\pi x}{L_x}\right) \cos\left(\frac{m\pi y}{L_y}\right)$$

where $C$ is determined by the initial average temperature.

## Applications

1. **Distributed heating systems**: Spatially patterned heat sources
2. **Semiconductor manufacturing**: Temperature control with localized heating/cooling
3. **Chemical reactors**: Exothermic/endothermic reactions with spatial structure
4. **Geothermal systems**: Periodic subsurface heat sources

## Visual Dynamics

- Initial condition diffuses while the source term maintains a persistent pattern
- Higher mode numbers ($n$, $m$) create finer spatial oscillations
- The solution approaches a steady state that balances diffusion and source

## References

- VisualPDE: [Inhomogeneous Heat Equation](https://visualpde.com/basic-pdes/inhomogeneous-heat-equation)
- Carslaw, H.S. & Jaeger, J.C. (1959). *Conduction of Heat in Solids*

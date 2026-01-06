# Inhomogeneous Wave Equation

## Mathematical Formulation

The inhomogeneous wave equation features spatially varying wave speed:

$$\frac{\partial^2 u}{\partial t^2} = \nabla \cdot (f(x,y) \nabla u)$$

where the diffusivity $f(x,y)$ is given by:

$$f(x,y) = D \left[1 + E \sin\left(\frac{m\pi x}{L_x}\right)\right] \left[1 + E \sin\left(\frac{n\pi y}{L_y}\right)\right]$$

For a square domain with $L = L_x = L_y$:

$$f(x,y) = D \left[1 + E \sin\left(\frac{m\pi x}{L}\right)\right] \left[1 + E \sin\left(\frac{n\pi y}{L}\right)\right]$$

### First-Order System

Converted to a first-order system for numerical integration:

$$\frac{\partial u}{\partial t} = v$$
$$\frac{\partial v}{\partial t} = \nabla \cdot (f(x,y) \nabla u)$$

### Expanded Form

The divergence expands to:

$$\nabla \cdot (f \nabla u) = f \nabla^2 u + \nabla f \cdot \nabla u$$

$$= f \nabla^2 u + \frac{\partial f}{\partial x}\frac{\partial u}{\partial x} + \frac{\partial f}{\partial y}\frac{\partial u}{\partial y}$$

## Physical Background

This equation models wave propagation in an inhomogeneous medium:

- The local wave speed is $c(x,y) = \sqrt{f(x,y)}$
- Regions with larger $f$ have faster wave propagation
- The wave refracts as it passes through regions of different wave speed

### Key Properties

1. **Variable Wave Speed**: The effective wave speed varies spatially according to $\sqrt{f(x,y)}$
2. **Wave Refraction**: Waves bend toward regions of slower wave speed (Snell's law analog)
3. **Energy Conservation**: Total wave energy is conserved (no damping in this formulation)
4. **Constraint**: Requires $|E| < 1$ to ensure $f(x,y) > 0$ everywhere

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Base diffusivity | $D$ | Base wave speed squared | 0.1 - 2.0 |
| Amplitude | $E$ | Strength of spatial variation | -0.9 to 0.9 |
| Mode number (x) | $n$ | Spatial frequency in x direction | 1 - 5 |
| Mode number (y) | $m$ | Spatial frequency in y direction | 1 - 5 |

## Boundary Conditions

This PDE is typically used with **homogeneous Neumann (no-flux) boundary conditions**:

$$\frac{\partial u}{\partial n}\bigg|_{\partial\Omega} = 0$$

This corresponds to waves reflecting off the boundaries.

## Physical Analogs

1. **Seismic waves**: Propagation through Earth with varying density/elasticity
2. **Light in graded-index media**: Optical fibers with varying refractive index
3. **Sound in atmosphere**: Acoustic propagation with temperature/density gradients
4. **Water waves**: Shallow water waves over varying depth

## Visual Dynamics

- Waves travel faster through regions where $f(x,y)$ is larger
- Watch for refraction effects as waves curve toward slower regions
- Higher mode numbers ($n$, $m$) create more complex spatial variations
- The parameter $E$ controls the contrast between fast and slow regions

## References

- VisualPDE: [Inhomogeneous Wave Equation](https://visualpde.com/basic-pdes/inhomogeneous-wave-equation)
- Morse, P.M. & Ingard, K.U. (1968). *Theoretical Acoustics*
- Born, M. & Wolf, E. (1999). *Principles of Optics*

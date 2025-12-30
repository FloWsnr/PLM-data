# Biharmonic Plate Equation

## Mathematical Formulation

The biharmonic equation (plate equation) describes fourth-order diffusion:

$$\frac{\partial u}{\partial t} = -D \nabla^4 u$$

where $\nabla^4 = \nabla^2(\nabla^2)$ is the biharmonic operator (bilaplacian):

$$\nabla^4 u = \frac{\partial^4 u}{\partial x^4} + 2\frac{\partial^4 u}{\partial x^2 \partial y^2} + \frac{\partial^4 u}{\partial y^4}$$

## Physical Background

The biharmonic equation arises in thin plate theory (Kirchhoff-Love plate theory):

- **Plate deflection**: Bending of thin elastic plates under load
- **Stream function**: In Stokes flow (very viscous fluids)
- **Higher-order smoothing**: More aggressive than standard diffusion

Key properties:
- Fourth-order spatial derivatives
- Extremely fast smoothing of high-frequency components
- Stiff equation requiring implicit solvers

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Biharmonic coefficient | $D$ | Bending stiffness / diffusion rate | 0.0001 - 1 |

## Comparison with Heat Equation

| Property | Heat Equation | Plate Equation |
|----------|---------------|----------------|
| Order | 2nd spatial | 4th spatial |
| Smoothing | Gaussian | More aggressive |
| Stiffness | Moderate | High |
| Solver | Explicit OK | Implicit needed |

## Applications

1. **Structural mechanics**: Thin plate and shell vibrations
2. **Fluid mechanics**: Stokes flow stream function
3. **Image processing**: Super-smooth denoising
4. **Computer graphics**: Surface fairing
5. **Elasticity**: Biharmonic stress functions

## Boundary Conditions

Plate problems typically require two boundary conditions at each boundary:
- **Clamped**: $u = 0$ and $\partial u/\partial n = 0$
- **Simply supported**: $u = 0$ and $\nabla^2 u = 0$
- **Free**: Moment and shear force conditions

## Numerical Considerations

- **Stiff equation**: Time step restrictions are severe for explicit methods
- **Implicit solver required**: Use implicit or semi-implicit schemes
- **Discretization**: Nine-point stencil for $\nabla^4$ in 2D
- **Stability**: $\Delta t \propto (\Delta x)^4$ for explicit methods

## References

- Timoshenko, S. & Woinowsky-Krieger, S. (1959). *Theory of Plates and Shells*
- Pozrikidis, C. (2014). *Introduction to Finite and Spectral Element Methods*

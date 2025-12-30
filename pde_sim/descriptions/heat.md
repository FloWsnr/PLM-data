# Heat Equation (Diffusion Equation)

## Mathematical Formulation

The heat equation is one of the most fundamental partial differential equations in mathematical physics:

$$\frac{\partial T}{\partial t} = D \nabla^2 T$$

where:
- $T$ is the temperature (or concentration in diffusion contexts)
- $D$ is the diffusion coefficient (thermal diffusivity)
- $\nabla^2$ is the Laplacian operator

In two dimensions, the Laplacian expands to:

$$\frac{\partial T}{\partial t} = D \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right)$$

## Physical Background

The heat equation was first derived by Joseph Fourier in 1822 to describe heat conduction in solid bodies. It belongs to the class of parabolic PDEs and is characterized by:

- **Smoothing behavior**: Initial discontinuities are immediately smoothed out
- **No finite propagation speed**: Information propagates instantly (unlike wave equations)
- **Energy dissipation**: Total "energy" (integral of $T^2$) decreases over time
- **Maximum principle**: Maximum and minimum values occur on boundaries or at initial time

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Diffusion coefficient | $D$ | Rate of heat/mass transport | 0.001 - 100 |

## Applications

1. **Thermal conduction**: Heat flow in metals, buildings, geological formations
2. **Molecular diffusion**: Spreading of chemicals in solutions
3. **Financial mathematics**: Black-Scholes equation is a variant
4. **Image processing**: Gaussian blurring corresponds to heat equation evolution
5. **Probability theory**: Connection to Brownian motion and random walks

## Analytical Solutions

For a Gaussian initial condition on an infinite domain:
$$T(x,y,t) = \frac{1}{4\pi Dt} \exp\left(-\frac{x^2 + y^2}{4Dt}\right)$$

## Numerical Considerations

- Explicit schemes require $\Delta t \leq \frac{(\Delta x)^2}{4D}$ for stability (CFL condition)
- Implicit schemes are unconditionally stable but require matrix inversions
- The equation is not stiff and can be solved efficiently with explicit methods for moderate $D$

## References

- Fourier, J. (1822). *Théorie analytique de la chaleur*
- Evans, L.C. (2010). *Partial Differential Equations*

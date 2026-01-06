# Heat Equation

The heat equation is a fundamental parabolic partial differential equation that describes how heat diffuses through a material over time.

## Description

The heat equation is a direct consequence of Fourier's law of heat conduction and the conservation of energy. Fourier's law states that heat flux is proportional to the negative temperature gradient - heat flows from hotter regions to colder regions at a rate proportional to the temperature difference.

This equation is one of the most important PDEs in mathematical physics and has applications far beyond thermal diffusion:

- **Thermal engineering**: Predicting temperature distributions in materials, thermal management in electronics
- **Financial mathematics**: The Black-Scholes equation for option pricing transforms into the heat equation
- **Image processing**: Gaussian blur and scale-space analysis use heat equation smoothing
- **Polymer science**: Measuring thermal diffusivity in rubber and polymeric materials
- **Porous media**: Pressure diffusion in porous materials follows the same equation
- **Biophysics**: Protein energy transfer and thermal modeling

A key property of solutions is the gradual smoothing of the initial temperature distribution. Any sharp features in the initial condition will immediately begin to diffuse, with high-frequency components decaying faster than low-frequency ones. This leads to the characteristic smoothing behavior where heat spreads from concentrated regions throughout the domain.

The heat equation exhibits infinite propagation speed - mathematically, a localized disturbance instantly affects all points in the domain, though the effect becomes negligible at large distances.

## Equations

The heat equation in two dimensions:

$$\frac{\partial T}{\partial t} = D_T \nabla^2 T$$

where:
- $T(x, y, t)$ is the temperature field
- $D_T$ is the thermal diffusivity (diffusion coefficient)
- $\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$ is the Laplacian operator

In the visual-pde framework, this is implemented as:
- Reaction term: `0` (no reaction/source terms)
- Diffusion: `D_T = 1` (diffusion coefficient for temperature field)

## Default Config

```yaml
solver: euler
dt: 0.01
dx: 1.5
domain_size: 320

boundary_x: neumann
boundary_y: neumann

species:
  - name: T
    diffusion: 1.0
```

## Parameter Variants

### heatEquation (2D)
The standard 2D heat equation with:
- No-flux (Neumann) boundary conditions
- Zero initial condition (add heat by clicking)
- Diffusion coefficient $D_T = 1$

### heatEquation1D
One-dimensional version with:
- Initial condition: $T(x,0) = \cos(m\pi x / L)$
- Parameter: `m = 8` in range `[1, 10]` (wave number)
- Demonstrates exponential decay of Fourier modes

The analytical solution is:
$$T(x,t) = e^{-D_T t (m\pi/L)^2} \cos\left(\frac{m\pi x}{L}\right)$$

Higher frequency modes (larger $m$) decay faster.

### heatEquation1DValidity
Validation preset comparing numerical solution to the analytical fundamental solution:
$$T(x,t) = \frac{1}{\sqrt{t+t_0}} \exp\left(-\frac{(x-L_x/2)^2}{4(t+t_0)}\right)$$

## References

- [Heat equation - Wikipedia](https://en.wikipedia.org/wiki/Heat_equation)
- [The Heat Equation - Paul's Online Math Notes](https://tutorial.math.lamar.edu/classes/de/theheatequation.aspx)
- [PDEs, Separation of Variables, and The Heat Equation - Mathematics LibreTexts](https://math.libretexts.org/Bookshelves/Differential_Equations/Differential_Equations_for_Engineers_(Lebl)/4:_Fourier_series_and_PDEs/4.06:_PDEs_separation_of_variables_and_the_heat_equation)

# Inhomogeneous Heat Equation

The inhomogeneous heat equation extends the standard heat equation with spatially-varying source/sink terms and diffusion coefficients, modeling systems with external heat input or heterogeneous materials.

## Description

The inhomogeneous (or nonhomogeneous) heat equation models thermal problems where heat is added or removed by external sources, or where the material properties vary in space. This is essential for realistic modeling of:

- **Heated rooms**: Temperature distribution with a heater switched on
- **Electronic devices**: Chips generating heat that diffuses through a substrate
- **Manufacturing processes**: Laser heating, welding, heat treatment
- **Geothermal systems**: Heat sources deep in the Earth's crust
- **Biological systems**: Metabolic heat generation in tissues
- **Composite materials**: Heat flow through materials with varying thermal conductivity

### Steady State Behavior

A key feature of the inhomogeneous heat equation is its steady state solution. When $\partial T/\partial t = 0$, the heat equation reduces to Poisson's equation:
$$D \nabla^2 T = -f(x,y)$$

**Important constraint**: For Neumann (no-flux) boundary conditions, the source term must satisfy:
$$\int_0^{L_y}\int_0^{L_x} f(x,y) \, dx \, dy = 0$$

If this integral is nonzero, the total heat in the domain grows or decays without bound - there's a net source or sink of energy with no way for it to escape.

### Spatially-Varying Diffusion

The more general form $\nabla \cdot (g(x,y) \nabla T)$ models materials where thermal conductivity varies in space. This creates interesting phenomena:
- Heat concentrates in low-diffusivity regions (slow to escape)
- Heat spreads quickly through high-diffusivity regions
- Sharp interfaces between regions create temperature discontinuities in the gradient

## Equations

### Forced Heat Equation (with source term)

$$\frac{\partial T}{\partial t} = D \nabla^2 T + f(x,y)$$

where:
- $T(x, y, t)$ is the temperature field
- $D$ is the diffusion coefficient
- $f(x,y)$ is the source/sink term

The preset uses:
$$f(x,y) = D\pi^2\left(\frac{n^2}{L_x^2} + \frac{m^2}{L_y^2}\right)\cos\left(\frac{n\pi x}{L_x}\right)\cos\left(\frac{m\pi y}{L_y}\right)$$

This specific form is chosen so the steady state solution is:
$$T(x,y) = -\cos\left(\frac{n\pi x}{L_x}\right)\cos\left(\frac{m\pi y}{L_y}\right)$$

Implementation:
- Reaction term: `cos(n*pi*x/L_x)*cos(m*pi*y/L_y)*pi^2*((n/L_x)^2 + (m/L_y)^2)`
- Diffusion: `D = 1`

### Heat Equation with Inhomogeneous Diffusion

$$\frac{\partial T}{\partial t} = \nabla \cdot (g(x,y) \nabla T)$$

where $g(x,y) > 0$ is the spatially-varying diffusion coefficient.

The preset uses:
$$g(x,y) = D\left[1 + E\cos\left(\frac{n\pi}{L_xL_y}\sqrt{(x-L_x/2)^2+(y-L_y/2)^2}\right)\right]$$

This creates radially-oscillating regions of high and low diffusion centered on the domain.

Implementation:
- Diffusion: `D*(1+E*cos(n*pi*sqrt((x-L_x/2)^2+(y-L_y/2)^2)/sqrt(L_x*L_y)))`
- Reaction term: `0`

## Default Config

### Forced Heat Equation (inhomogHeatEquation)
```yaml
solver: euler
dt: 0.004
dx: 0.5
domain_size: 100

boundary_x: neumann
boundary_y: neumann

parameters:
  n: 4    # range: [1, 10] - x-direction wave number
  m: 4    # range: [1, 10] - y-direction wave number

species:
  - name: T
    diffusion: 1.0
```

### Inhomogeneous Diffusion (inhomogDiffusionHeatEquation)
```yaml
solver: euler
dt: 0.004
dx: 0.5
domain_size: 100

boundary_x: dirichlet
boundary_y: dirichlet

parameters:
  D: 1.0      # base diffusion coefficient
  E: 0.99     # range: [0, 1] - modulation amplitude
  n: 40       # range: [1, 50] - number of radial oscillations

species:
  - name: T
    diffusion: D*(1+E*cos(...))  # spatially varying
```

## Parameter Variants

### inhomogHeatEquation
Forced heat equation with sinusoidal source term:
- Source term creates regions of heat addition and removal
- Parameters `n` and `m` control spatial frequency of forcing
- Neumann boundaries ensure conservation when source integrates to zero
- Steady state is a cosine pattern matching the forcing frequency
- Brush disabled (interaction would disrupt convergence to steady state)

### inhomogDiffusionHeatEquation
Heat equation with spatially-varying diffusion coefficient:
- Radially-oscillating diffusion coefficient centered on domain
- Initial condition: $T(x,y,0) = 1$ (uniform)
- Dirichlet boundaries: $T = 0$ at edges
- Parameter `E` controls contrast between high/low diffusion regions (E close to 1 gives maximum contrast)
- Parameter `n` controls number of radial rings
- Heat partitions into regions bounded by diffusion maxima
- Creates striking banded patterns as heat escapes preferentially through high-diffusion channels

## References

- [The Nonhomogeneous Heat Equation - Mathematics LibreTexts](https://math.libretexts.org/Bookshelves/Differential_Equations/Introduction_to_Partial_Differential_Equations_(Herman)/07:_Green's_Functions/7.03:_The_Nonhomogeneous_Heat_Equation)
- [Heat equation - Wikipedia](https://en.wikipedia.org/wiki/Heat_equation)
- [Nonhomogeneous Problems - Portland State University](https://web.pdx.edu/~daescu/mth427_527/notes_weeks6-7.pdf)
- [Heat with a source - UC Santa Barbara](https://web.math.ucsb.edu/~grigoryan/124A/lecs/lec15.pdf)

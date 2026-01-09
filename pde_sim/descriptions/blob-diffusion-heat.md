# Blob Diffusion Heat Equation

Heat equation with randomly distributed gaussian blobs of varying thermal conductivity.

## Description

This variant of the heat equation models heat conduction through materials with randomly distributed inclusions or heterogeneities. Unlike the radial pattern in `inhomogeneous-diffusion-heat`, here the **diffusion coefficient consists of gaussian blobs** at random positions.

The spatial variation creates isolated regions of high thermal conductivity (at blob centers) surrounded by low conductivity background. Heat flows preferentially through the high-conductivity blob regions.

### Physical Applications

- **Composite materials**: Heat flow through materials with embedded particles
- **Porous media**: Varying permeability in rock or soil
- **Biological tissue**: Heterogeneous thermal properties (e.g., tumors, blood vessels)
- **Metallurgy**: Heat treatment of materials with inclusions
- **Geothermal**: Heat flow through heterogeneous geological formations

## Equations

The governing equation with spatially-varying diffusion:

$$\frac{\partial T}{\partial t} = \nabla \cdot (g(x,y) \nabla T)$$

where the diffusion coefficient is a sum of gaussian blobs:

$$g(x,y) = D_{min} + \sum_{i=1}^{n} (D_{max} - D_{min}) \exp\left(-\frac{(x-x_i)^2 + (y-y_i)^2}{2\sigma^2}\right)$$

The blob centers $(x_i, y_i)$ are randomly distributed across the domain.

Expanding the divergence using the product rule:

$$\frac{\partial T}{\partial t} = g \nabla^2 T + \frac{\partial g}{\partial x}\frac{\partial T}{\partial x} + \frac{\partial g}{\partial y}\frac{\partial T}{\partial y}$$

## Parameter Details

- **D_min**: Minimum (background) diffusion coefficient. Controls the baseline heat transfer rate.

- **D_max**: Maximum diffusion coefficient at blob centers. Higher values create stronger channeling of heat through blobs.

- **n_blobs**: Number of gaussian blobs. More blobs create more complex diffusion patterns.

- **sigma**: Blob size as fraction of domain (0.02-0.3). Controls the spatial extent of high-conductivity regions.

- **seed**: Random seed for reproducible blob placement.

## Parameter Variants

### Default (Sparse Blobs)
- `D_min = 0.1`, `D_max = 2.0`
- `n_blobs = 5`, `sigma = 0.1`
- Creates isolated high-conductivity regions

### Dense Blobs
- `D_min = 0.2`, `D_max = 1.5`
- `n_blobs = 15`, `sigma = 0.08`
- More uniform but still heterogeneous

### Strong Contrast
- `D_min = 0.05`, `D_max = 3.0`
- `n_blobs = 7`, `sigma = 0.12`
- Very pronounced channeling effect

## Behavior

Heat flows preferentially through blob regions (high conductivity). Starting from localized heat sources:

1. **Channeling**: Heat "finds" nearby blobs and flows through them faster
2. **Trapping**: Heat in low-conductivity regions diffuses slowly
3. **Complex patterns**: Interaction between blobs creates intricate temperature distributions

With periodic boundaries, heat eventually equilibrates. With Dirichlet boundaries, heat drains preferentially through blobs near boundaries.

## References

- [Heat equation with variable coefficients](https://en.wikipedia.org/wiki/Heat_equation#Inhomogeneous_equation)
- [VisualPDE Inhomogeneous Diffusion](https://visualpde.com/basic-pdes/inhomogeneous-diffusion)

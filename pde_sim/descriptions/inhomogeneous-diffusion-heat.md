# Inhomogeneous Diffusion Heat Equation

The heat equation with spatially-varying thermal conductivity, creating regions of fast and slow heat diffusion.

## Description

This variant of the heat equation models heat conduction through materials with non-uniform thermal properties. Unlike the standard inhomogeneous heat equation (which has a spatially-varying source term), here the **diffusion coefficient itself varies in space**.

The spatial variation creates concentric rings of alternating high and low thermal conductivity, centered on the domain. Heat partitions into bands bounded by conductivity maxima, as diffusion naturally flows from high to low conductivity regions.

### Physical Applications

- **Composite materials**: Heat flow through layered or graded materials
- **Geological modeling**: Heat conduction through heterogeneous rock formations
- **Manufacturing**: Welding and heat treatment of non-uniform materials
- **Biological tissue**: Heat transfer through tissues with varying thermal properties
- **Thermal management**: Design of heat spreaders with engineered conductivity

## Equations

The governing equation with spatially-varying diffusion:

$$\frac{\partial T}{\partial t} = \nabla \cdot (g(x,y) \nabla T)$$

where the diffusion coefficient has radial variation from the domain center:

$$g(x,y) = D \left(1 + E \cos\left(\frac{n\pi r}{\sqrt{L_x L_y}}\right)\right)$$

with radial distance:

$$r = \sqrt{(x - L_x/2)^2 + (y - L_y/2)^2}$$

Expanding the divergence using the product rule:

$$\frac{\partial T}{\partial t} = g \nabla^2 T + \frac{\partial g}{\partial x}\frac{\partial T}{\partial x} + \frac{\partial g}{\partial y}\frac{\partial T}{\partial y}$$

The gradient terms account for heat flow induced by conductivity gradients, not just temperature gradients.

## Default Config

```yaml
solver: euler
dt: 0.01
dx: 0.5
domain_size: 100

boundary_x: dirichlet
boundary_y: dirichlet

parameters:
  D: 1.0     # base diffusion coefficient
  E: 0.99    # modulation amplitude [0, 1)
  n: 40      # number of radial oscillations [1, 50]

initial_condition: T = 1 (uniform)

species:
  - name: T  # temperature
```

## Parameter Details

- **D**: Base diffusion coefficient. Controls the overall rate of heat transfer.

- **E**: Modulation amplitude (0 ≤ E < 1). Controls the contrast between high and low conductivity regions:
  - E = 0: Uniform conductivity (standard heat equation)
  - E close to 1: Strong contrast, pronounced banding patterns
  - The diffusion coefficient ranges from D(1-E) to D(1+E)

- **n**: Number of radial oscillations. Controls the fineness of the conductivity pattern:
  - Low n (1-10): Few, wide bands
  - High n (30-50): Many, narrow bands
  - Creates n complete oscillation cycles from center to corner

## Parameter Variants

### Fine Radial Pattern (Default)
High-frequency radial oscillations:
- `E = 0.99`: Strong conductivity contrast
- `n = 40`: Many narrow rings
- Creates detailed banding patterns

### Coarse Radial Pattern
Fewer, wider conductivity bands:
- `E = 0.9`
- `n = 5`
- Easier to visualize, faster dynamics

### Weak Inhomogeneity
Nearly uniform with slight variation:
- `E = 0.3`
- `n = 10`
- Perturbative regime, small deviations from standard heat equation

## Behavior

Starting from a uniform temperature (T = 1), heat flows out through the Dirichlet boundaries (T = 0). The spatially-varying conductivity causes:

1. **Radial banding**: Temperature "sticks" at conductivity maxima
2. **Non-uniform decay**: Regions with high g cool faster
3. **Pattern formation**: Concentric rings of temperature persist longer than in uniform media

The steady state is T = 0 everywhere (Dirichlet boundary conditions drain all heat), but the transient evolution shows interesting spatial structure.

## References

- [Heat equation with variable coefficients](https://en.wikipedia.org/wiki/Heat_equation#Inhomogeneous_equation)
- [VisualPDE Inhomogeneous Diffusion](https://visualpde.com/basic-pdes/inhomogeneous-diffusion)
- Carslaw, H.S. & Jaeger, J.C. (1959). "Conduction of Heat in Solids" - Oxford University Press

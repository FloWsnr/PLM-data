# Darcy Flow Equation

The Darcy flow equation describes pressure-driven flow through porous media, combining Darcy's law with mass conservation to model how fluid moves through saturated porous materials.

## Description

Darcy's law, established experimentally by Henry Darcy in 1856, states that the volumetric flow rate through a porous medium is proportional to the pressure gradient and inversely proportional to the fluid viscosity. When combined with the continuity equation (conservation of mass), this yields a parabolic PDE for the pressure field.

The Darcy flow equation is fundamental to many applications:

- **Groundwater hydrology**: Modeling aquifer flow, well drawdown, and contaminant transport
- **Petroleum engineering**: Reservoir simulation for oil and gas extraction
- **Filtration processes**: Design of filters, membranes, and separation systems
- **Soil mechanics**: Consolidation of saturated soils under loading
- **Biomedical engineering**: Flow through biological tissues and scaffolds

The equation assumes:
- Laminar flow (valid for low Reynolds numbers typical in porous media)
- Saturated conditions (pores completely filled with fluid)
- Incompressible fluid
- Rigid porous matrix (no deformation)

Key behaviors:
- Without sources (f = 0), pressure diffuses toward a uniform equilibrium
- Sources (f > 0) create local pressure maxima that drive outward flow
- Sinks (f < 0) create local pressure minima that draw in surrounding fluid
- The permeability K controls the rate of pressure equilibration

## Equations

The Darcy flow equation:

$$\frac{\partial p}{\partial t} = K \nabla^2 p + f$$

where:
- $p(x, t)$ is the pressure field [Pa]
- $K$ is the permeability/hydraulic conductivity [m^2/s]
- $f$ is the source/sink term [Pa/s]
- $\nabla^2$ is the Laplacian operator

The permeability $K$ combines material and fluid properties:
$$K = \frac{k}{\mu}$$
where $k$ is the intrinsic permeability [m^2] and $\mu$ is the dynamic viscosity [Pa*s].

Darcy velocity (superficial velocity) is obtained from:
$$\mathbf{q} = -K \nabla p$$

## Default Config

```yaml
preset: darcy
parameters:
  K: 1.0   # Permeability/hydraulic conductivity
  f: 0.0   # Source/sink term

init:
  type: gaussian-blob
  params:
    amplitude: 1.0
    sigma: 0.15

solver: euler
dt: 0.001
t_end: 1.0

resolution: [64, 64]
domain_size: [1.0, 1.0]

bc:
  x-: neumann:0
  x+: neumann:0
  y-: neumann:0
  y+: neumann:0

output:
  formats: [png]
  num_frames: 100
```

## Parameter Variants

### No-flux boundaries (impermeable walls)
Standard configuration with Neumann (no-flux) boundaries representing impermeable container walls:
```yaml
bc:
  x-: neumann:0
  x+: neumann:0
  y-: neumann:0
  y+: neumann:0
```

### Fixed pressure boundaries (reservoirs)
Dirichlet conditions representing constant-pressure reservoirs:
```yaml
bc:
  x-: dirichlet:1.0   # High pressure reservoir
  x+: dirichlet:0.0   # Low pressure outlet
  y-: neumann:0
  y+: neumann:0
```

### With injection source
Simulating fluid injection:
```yaml
parameters:
  K: 1.0
  f: 0.5  # Positive source = injection
```

### With production sink
Simulating fluid extraction (e.g., pumping well):
```yaml
parameters:
  K: 1.0
  f: -0.5  # Negative = sink/extraction
```

### Low permeability (tight formation)
```yaml
parameters:
  K: 0.1
  f: 0.0
```

### High permeability (coarse sand/gravel)
```yaml
parameters:
  K: 10.0
  f: 0.0
```

## References

- [Darcy's law - Wikipedia](https://en.wikipedia.org/wiki/Darcy%27s_law)
- [Groundwater flow equation - Wikipedia](https://en.wikipedia.org/wiki/Groundwater_flow_equation)
- Bear, J. (1972). Dynamics of Fluids in Porous Media. American Elsevier.
- Freeze, R.A. and Cherry, J.A. (1979). Groundwater. Prentice-Hall.

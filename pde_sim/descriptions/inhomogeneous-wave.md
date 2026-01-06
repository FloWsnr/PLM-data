# Inhomogeneous Wave Equation

The inhomogeneous wave equation models wave propagation through media with spatially-varying properties, where wave speed depends on position.

## Description

The inhomogeneous wave equation extends the standard wave equation to handle materials with non-uniform properties. Instead of a constant wave speed, the propagation velocity varies across the domain. This is crucial for modeling:

- **Seismology**: Earthquake waves traveling through geological layers with different densities and elastic properties
- **Acoustics**: Sound propagation in rooms with varying temperature or through composite materials
- **Optics**: Light propagation in graded-index optical fibers or through atmospheric layers
- **Ocean acoustics**: Sound waves traveling through water layers with varying temperature and salinity
- **Ultrasonics**: Medical imaging through heterogeneous tissue
- **Metamaterials**: Engineered materials with designed spatially-varying properties

### Physical Phenomena

When waves encounter regions of varying wave speed:
- **Refraction**: Waves bend toward regions of lower wave speed
- **Reflection**: Partial reflection occurs at interfaces between different media
- **Focusing/Defocusing**: Curved interfaces or gradual changes can act as lenses
- **Trapping**: Waves can become trapped in low-speed regions (waveguides)

The simulation demonstrates how waves slow down in regions where $f(x,y)$ is smaller, creating visually apparent delays and distortions in the wave pattern.

### Damped Waves

The damped wave equation adds energy dissipation:
$$\frac{\partial^2 u}{\partial t^2} + d\frac{\partial u}{\partial t} = D\nabla^2 u$$

The damping term $d\frac{\partial u}{\partial t}$ causes wave amplitude to decay over time, modeling:
- Friction and viscosity in mechanical systems
- Electrical resistance in transmission lines
- Absorption in acoustic media

## Equations

### Inhomogeneous Wave Equation

$$\frac{\partial^2 u}{\partial t^2} = \nabla \cdot (f(x,y) \nabla u)$$

where:
- $u(x, y, t)$ is the displacement/pressure field
- $f(x,y) > 0$ is the spatially-varying wave speed squared

The preset uses:
$$f(x,y) = D\left[1 + E\sin\left(\frac{m\pi x}{L_x}\right)\right]\left[1 + E\sin\left(\frac{n\pi y}{L_y}\right)\right]$$

This creates a grid of regions with alternating high and low wave speeds. Corners where both sine terms are negative have the slowest wave propagation.

Reformulated as first-order system:
- $\frac{\partial u}{\partial t} = v + CD\nabla^2 u$
- $\frac{\partial v}{\partial t} = \nabla \cdot (f(x,y) \nabla u)$

Implementation:
- Reaction terms: `u: v`, `v: 0`
- Diffusion: `u-u: C*D`, `v-u: D*(1+E*sin(pi*m*x/L_x))*(1+E*sin(n*pi*y/L_y))`

### Damped Wave Equation

$$\frac{\partial^2 u}{\partial t^2} + d\frac{\partial u}{\partial t} = D\nabla^2 u$$

Reformulated:
- $\frac{\partial u}{\partial t} = v + CD\nabla^2 u$
- $\frac{\partial v}{\partial t} = D\nabla^2 u - dv$

Implementation:
- Reaction terms: `u: v`, `v: -d*v`
- Diffusion: `u-u: C*D`, `v-u: D`

## Default Config

### Inhomogeneous Wave (inhomogWaveEquation)
```yaml
solver: euler
dt: 0.001
dx: 0.3
domain_size: 100

boundary_x: neumann
boundary_y: neumann

parameters:
  D: 1.0     # base wave speed squared
  m: 9       # range: [0, 10] - x-direction wave number
  n: 9       # range: [0, 10] - y-direction wave number
  C: 0.01    # damping/stabilization coefficient
  E: 0.97    # modulation amplitude (must be < 1)

species:
  - name: u  # displacement
  - name: v  # velocity
```

### Damped Wave (dampedWaveEquation)
```yaml
solver: euler
dt: 0.002
dx: 0.5
domain_size: 100

boundary_x: dirichlet
boundary_y: dirichlet

dirichlet_value: cos(m*x*pi/100)*cos(m*y*pi/100)

parameters:
  D: 1.0     # wave speed squared
  C: 0.01    # numerical stabilization
  m: 8       # range: [0, 10] - boundary forcing frequency
  d: 0       # range: [0, 0.1], step: 0.01 - damping coefficient

species:
  - name: u
  - name: v
```

## Parameter Variants

### inhomogWaveEquation
Wave equation with spatially-varying wave speed:
- Sinusoidal modulation of wave speed in x and y directions
- Parameter `E = 0.97` gives strong contrast (wave speed varies by factor ~60)
- Corners of the modulation pattern have very slow wave propagation
- `m` and `n` control the spatial frequency of the wave speed pattern
- Neumann boundary conditions

### dampedWaveEquation
Damped wave equation with oscillating boundary conditions:
- Dirichlet boundary conditions with cosine pattern
- Waves propagate from boundaries into initially quiet domain
- Parameter `d` controls damping (d=0 gives undamped waves)
- Parameter `m` controls frequency of boundary forcing
- Square domain enforced for symmetric patterns
- Initial condition: $u = 0$ everywhere

### wavesAddedGeometry
Wave equation with internal obstacles:
- Uses an indicator function to define internal boundaries
- Third species `w` marks domain regions (w < 0.5 indicates obstacle)
- Users can paint obstacles that deflect waves
- Initial condition: localized pulse that interacts with user-drawn obstacles
- 3D surface visualization available

## References

- [Wave equation - Wikipedia](https://en.wikipedia.org/wiki/Wave_equation)
- [Inhomogeneous wave equation - VisualPDE](https://visualpde.com/basic-pdes/inhomogeneous-wave-equation.html)

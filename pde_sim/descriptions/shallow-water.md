# Shallow Water Equations

The shallow water equations model water waves and ripples in a fluid layer where horizontal length scales are much larger than the vertical depth.

## Description

The shallow water equations are fundamental in oceanography, coastal engineering, and atmospheric science. They describe the dynamics of a thin layer of fluid under gravity, capturing phenomena like tsunamis, tidal waves, storm surges, and dam break floods.

The equations couple the water height h with horizontal velocity components u and v. Key physical effects include:
- Gravity waves propagating at speed sqrt(g*H) where H is the mean depth
- Coriolis force from Earth's rotation (parameter f), crucial for large-scale geophysical flows
- Friction and dissipation (parameters k and epsilon)
- Nonlinear advection terms creating steepening waves and solitons

The Coriolis force leads to geostrophic balance in large-scale ocean currents, where pressure gradients balance rotational effects. This explains why different parts of the ocean can maintain different depths. The equations also exhibit Kelvin-Helmholtz (shear) instabilities when velocity gradients are present.

Dam break simulations demonstrate how the Coriolis force can stabilize wave fronts - a counterintuitive result important for understanding ocean dynamics. Vortical solitons emerge when f is significant, showing anticyclonic rotation (positive vorticity indicates counterclockwise flow in the Northern Hemisphere convention).

## Equations

Water height evolution:
$$\frac{\partial h}{\partial t} = -\left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right)(h + H_e) - \left(\frac{\partial h}{\partial x}u + \frac{\partial h}{\partial y}v\right) - \varepsilon h$$

Horizontal momentum (x-direction):
$$\frac{\partial u}{\partial t} = \nu \nabla^2 u - g\frac{\partial h}{\partial x} - ku - u\frac{\partial u}{\partial x} - v\frac{\partial u}{\partial y} + fv$$

Horizontal momentum (y-direction):
$$\frac{\partial v}{\partial t} = \nu \nabla^2 v - g\frac{\partial h}{\partial y} - kv - u\frac{\partial v}{\partial x} - v\frac{\partial v}{\partial y} - fu$$

## Default Config

```yaml
solver: euler
dt: 0.005
dx: 2
domain_size: 400

boundary_h: neumann
boundary_u: dirichlet = 0 (left/right), neumann (top/bottom)
boundary_v: neumann (left/right), dirichlet = 0 (top/bottom)

parameters:
  H_e: 1       # equilibrium water depth
  g: 9.81      # gravitational acceleration
  f: 0.01      # Coriolis parameter, range [0, 1]
  k: 0.001     # linear drag coefficient
  nu: 0.5      # kinematic viscosity
  epsilon: 0.0001  # height dissipation rate
```

## Parameter Variants

### ShallowWaterEqns (base)
Standard 2D shallow water with reflective boundaries. Clicking initiates point waves. Default view shows 3D surface plot of water height.

### ShallowWaterEqnsDamBreaking
Dam break simulation:
- Initial condition: step function in h (tanh profile)
- Periodic boundaries on all sides
- f = 0 by default (try f = 0.4 or f = 1 to see Coriolis stabilization)
- Reduced height diffusion (0.01)

### ShallowWaterEqnsVorticalSolitons
Geostrophically balanced vortices:
- f = 1 (strong Coriolis)
- Clicking places vortical solitons with positive vorticity
- 3D surface colored by vorticity
- Longer click creates deeper vortices

### 1DShallowWaterEqns
One-dimensional nonlinear solitary waves:
- Sech-squared initial conditions (soliton-like)
- domain_size: 1000
- Periodic boundaries
- No Coriolis (1D flow)

### 1DLinearizedShallowWaterEqns
Linearized 1D version for comparison:
- Simplified reaction terms without nonlinear advection
- Shows qualitative similarity but quantitative differences from nonlinear case

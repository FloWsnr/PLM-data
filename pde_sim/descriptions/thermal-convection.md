# Thermal Convection (Rayleigh-Benard)

Thermal convection in a 2D Boussinesq model, describing buoyancy-driven fluid motion from temperature gradients.

## Description

This system models Rayleigh-Benard convection, a fundamental phenomenon where a horizontal fluid layer heated from below becomes unstable and develops organized convection cells. First investigated by Joseph Valentin Boussinesq and Anton Oberbeck in the 19th century, this instability is ubiquitous in nature - from atmospheric thermals to mantle convection in Earth's interior.

The Boussinesq approximation treats density variations as negligible except in the buoyancy term, where temperature differences drive vertical motion. The key dimensionless parameters are:
- Rayleigh number (Ra): ratio of buoyancy forces to viscous dissipation and heat conduction
- Prandtl number (Pr = nu/kappa): ratio of momentum diffusivity to thermal diffusivity

When Ra exceeds a critical value (approximately 1707 for idealized boundaries), the system transitions from static conduction to convective motion. Small perturbations grow and organize into Benard cells - counter-rotating convection rolls that efficiently transport heat.

The vorticity-streamfunction formulation is used here, with the temperature field b representing the perturbation from the top boundary temperature. Heating at the bottom boundary (via T_b) drives the instability. The coupling term b_x in the vorticity equation represents horizontal buoyancy gradients driving circulation.

## Equations

Vorticity evolution with buoyancy forcing:
$$\frac{\partial \omega}{\partial t} = \nu \nabla^2 \omega - \frac{\partial \psi}{\partial y}\frac{\partial \omega}{\partial x} + \frac{\partial \psi}{\partial x}\frac{\partial \omega}{\partial y} + \frac{\partial b}{\partial x}$$

Stream function (relaxation form):
$$\varepsilon \frac{\partial \psi}{\partial t} = \nabla^2 \psi + \omega$$

Temperature perturbation:
$$\frac{\partial b}{\partial t} = \kappa \nabla^2 b - \left(\frac{\partial \psi}{\partial y}\frac{\partial b}{\partial x} - \frac{\partial \psi}{\partial x}\frac{\partial b}{\partial y}\right)$$

Velocity from stream function:
$$u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}$$

## Default Config

```yaml
solver: euler
dt: 0.001
dx: 1.5  # inherited from NavierStokesVorticity
domain_size: 500  # inherited from NavierStokesVorticity

boundary_omega: dirichlet = 0 (top/bottom), periodic (left/right)
boundary_psi: dirichlet = 0 (top/bottom), periodic (left/right)
boundary_b: neumann = T_b (bottom), dirichlet = 0 (top), periodic (left/right)

parameters:
  nu: 0.2      # kinematic viscosity, range [0.01, 1]
  epsilon: 0.05  # relaxation parameter for stream function
  D: 0.05      # passive scalar diffusion (for S field)
  kappa: 0.5   # thermal diffusivity
  T_b: 0.08    # bottom boundary temperature flux
```

## Parameter Variants

### thermalConvection (base)
Boundary-driven convection model. Temperature perturbation b is plotted by default. Heating at lower boundary becomes unstable to high-frequency perturbations that grow and coalesce into larger Rayleigh-Benard cells. Clicking adds warm air regions that convect upward.

### thermalConvectionInitialData
Larger initial temperature perturbations:
- Instability develops away from boundary in bulk fluid
- Same physics, different initial length scales

### thermalConvectionBoundaries
Modified boundary conditions:
- Temperature forcing at both top and bottom boundaries
- No-flux horizontal conditions (replacing periodic)
- Clearer visualization of convective cell formation and merging

### DurhamConvection
Artistic simulation with:
- Custom color schemes and blending
- Stochastic forcing
- Visual overlay of Durham cathedral

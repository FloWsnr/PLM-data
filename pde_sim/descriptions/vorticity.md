# Vorticity Equation (2D Incompressible Flow)

The vorticity-streamfunction formulation of 2D fluid dynamics, describing spinning fluid motion through vortex dynamics.

## Description

The vorticity equation reformulates the Navier-Stokes equations by focusing on vorticity (the curl of velocity) rather than velocity itself. This approach provides better physical insight for 2D incompressible flows, as noted by Chemin: "The key quantity for understanding 2D incompressible fluids is the vorticity."

In 2D, the vorticity reduces to a scalar field omega pointing perpendicular to the flow plane (out of screen when positive). The stream function psi automatically satisfies incompressibility and provides velocity through spatial derivatives. Contours of constant psi are streamlines - the paths followed by fluid particles.

Key physical insights:
- Vorticity is advected by the flow and diffused by viscosity
- In 2D, vortex stretching is absent (unlike 3D), making the dynamics simpler
- Following a fluid particle, angular momentum changes only through viscous diffusion
- Positive omega indicates counterclockwise rotation, negative indicates clockwise

The Poisson equation relating psi and omega is elliptic and cannot be solved directly by VisualPDE's timestepping. Instead, a parabolic relaxation with small parameter epsilon is used, which converges to the true Poisson solution.

A passive scalar field S demonstrates convection-diffusion behavior in the computed flow field, useful for visualizing flow patterns and mixing.

## Equations

Vorticity transport:
$$\frac{\partial \omega}{\partial t} = \nu \nabla^2 \omega - \frac{\partial \psi}{\partial y}\frac{\partial \omega}{\partial x} + \frac{\partial \psi}{\partial x}\frac{\partial \omega}{\partial y}$$

Stream function (parabolic relaxation of Poisson equation):
$$\varepsilon \frac{\partial \psi}{\partial t} = \nabla^2 \psi + \omega$$

Passive scalar transport:
$$\frac{\partial S}{\partial t} = D \nabla^2 S - \left(\frac{\partial \psi}{\partial y}\frac{\partial S}{\partial x} - \frac{\partial \psi}{\partial x}\frac{\partial S}{\partial y}\right)$$

Velocity from stream function:
$$u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}$$

Vorticity definition (for reference):
$$\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$$

## Default Config

```yaml
solver: euler
dt: 0.01
dx: 1.5
domain_size: 500

boundary_omega: periodic (all boundaries)
boundary_psi: periodic (all boundaries)
boundary_S: periodic (top/bottom), neumann = 0 (left/right)

parameters:
  nu: 0.05     # kinematic viscosity, range [0.01, 1]
  epsilon: 0.05  # relaxation parameter for Poisson solver
  D: 0.05      # passive scalar diffusion coefficient
```

## Initial Conditions

In VisualPDE's [interactive simulation](/sim/?preset=NavierStokesVorticity):
- omega: zero everywhere (users add vortices by clicking)
- psi: zero (solved via parabolic relaxation)
- S: gradient (L_x-x)/L_x for visualizing flow patterns

Our implementation uses a vortex-pair initial condition:
- omega: two counter-rotating Gaussian vortices
- psi: zero (relaxes to solution)
- S: gradient from 1 (left) to 0 (right)

## Parameter Variants

### NavierStokesVorticity (base)
Standard vorticity-streamfunction formulation:
- In VisualPDE: clicking adds positive vortices (counterclockwise), right-clicking adds negative
- Equal and opposite vortices nearby create approximately unidirectional flow
- Default view plots vorticity omega with diverging colormap
- Alternative views: u, v, speed sqrt(psi_x^2 + psi_y^2), passive scalar S

### NavierStokesVorticityBounded (not yet implemented)
Modified for bounded domain studies:
- Oscillatory initial vorticity with wavenumber k
- Initial condition: A*cos(k*pi*x/L_x)*cos(k*pi*y/L_x) where A = 0.005*k^1.5
- k = 51 by default, adjustable range [1, 100]
- D = 0.1 (vs base 0.05)
- Dirichlet omega = 0 on left/right boundaries, periodic top/bottom
- Demonstrates decay and reorganization of complex vorticity fields

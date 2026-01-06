# Navier-Stokes Equations

The 2D incompressible Navier-Stokes equations govern the motion of viscous fluids, describing how velocity and pressure fields evolve over time.

## Description

The Navier-Stokes equations are fundamental to fluid dynamics and can be seen as Newton's second law of motion for fluids. They model diverse phenomena including weather patterns, atmospheric flow, ocean currents, and pipe flow. The equations were derived by Stokes in England and Navier in France in the early 1800s as extensions of the Euler equations that include viscosity effects.

The incompressibility constraint requires that the velocity field be divergence-free, which physically means fluid volume is conserved. The Reynolds number Re = rho*U*L/mu characterizes the ratio of inertial to viscous forces - low Reynolds number flows are laminar while high Reynolds number flows become turbulent.

A passive scalar field S is included to track fluid motion through advection and diffusion. The vortex shedding simulation demonstrates how cylindrical obstacles create periodic vortex patterns behind them (the von Karman vortex street), a classic phenomenon in fluid dynamics.

The numerical implementation uses a generalized pressure equation approach, treating pressure evolution through a relaxation method with a tuning parameter M related to the Mach number and pressure wave speed.

## Equations

Momentum equations:
$$\frac{\partial u}{\partial t} = -\left(u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y}\right) - \frac{\partial p}{\partial x} + \nu \nabla^2 u$$

$$\frac{\partial v}{\partial t} = -\left(u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y}\right) - \frac{\partial p}{\partial y} + \nu \nabla^2 v$$

Generalized pressure equation:
$$\frac{\partial p}{\partial t} = \nu \nabla^2 p - \frac{1}{M^2}\left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right)$$

Passive scalar transport:
$$\frac{\partial S}{\partial t} = -\left(u \frac{\partial S}{\partial x} + v \frac{\partial S}{\partial y}\right) + D \nabla^2 S$$

## Default Config

```yaml
solver: euler
dt: 0.01
dx: 1
domain_size: 300

boundary_x_u: periodic (left/right)
boundary_y_u: dirichlet = 0 (top/bottom)
boundary_x_v: periodic (left/right)
boundary_y_v: dirichlet = 0 (top/bottom)
boundary_p: neumann
boundary_S: combo (gradient initial condition)

parameters:
  nu: 0.02  # kinematic viscosity, range [0.02, 20]
  M: 0.5    # Mach number parameter
  D: 0.05   # passive scalar diffusion coefficient
```

## Parameter Variants

### NavierStokes (base)
Standard velocity-pressure formulation with passive scalar tracking. Speed is plotted by default with velocity arrows.

### NavierStokesFlowCylinder
Modified for vortex shedding simulations:
- Clicking places cylindrical obstacles that the flow moves around
- Uses S field as obstruction indicator
- Additional parameters: U = 0.7 (inlet velocity), a = 0.05 (cylinder radius factor)
- dt: 0.03, dx: 2
- Includes momentum damping term: -u*max(S,0) in u equation, -v*max(S,0) in v equation

### NavierStokesPoiseuilleFlow
Channel flow configuration with pressure-driven flow between parallel plates.

# Compressible Navier-Stokes Equations

The 2D compressible Navier-Stokes equations model viscous fluid flow where density varies and pressure is governed by thermodynamics (ideal gas law).

## Description

Unlike the incompressible Navier-Stokes equations where density is constant and the velocity field is divergence-free, the compressible formulation allows density variations and captures acoustic phenomena, shock waves, and compressibility effects. These equations are fundamental to aerodynamics, astrophysical flows, and any regime where the Mach number is not negligibly small.

The system is written in primitive variables (density, velocity, pressure) rather than conservative variables, which is more natural for smooth flows and avoids the need for Riemann solvers. The pressure equation is derived from the energy equation for an ideal gas (p = rho * R * T), with an added thermal diffusion term for numerical stability.

The ratio of specific heats gamma characterizes the gas: gamma = 1.4 for diatomic gases (air, N2), gamma = 5/3 for monatomic gases (argon, helium). The dynamic viscosity mu controls momentum diffusion, while the thermal diffusivity kappa stabilizes the pressure equation by smoothing temperature gradients.

## Equations

Continuity (mass conservation):
$$\frac{\partial \rho}{\partial t} = -u \frac{\partial \rho}{\partial x} - v \frac{\partial \rho}{\partial y} - \rho \left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right)$$

x-momentum:
$$\frac{\partial u}{\partial t} = -u \frac{\partial u}{\partial x} - v \frac{\partial u}{\partial y} - \frac{1}{\rho} \frac{\partial p}{\partial x} + \frac{\mu}{\rho} \nabla^2 u$$

y-momentum:
$$\frac{\partial v}{\partial t} = -u \frac{\partial v}{\partial x} - v \frac{\partial v}{\partial y} - \frac{1}{\rho} \frac{\partial p}{\partial y} + \frac{\mu}{\rho} \nabla^2 v$$

Pressure (from energy equation for ideal gas):
$$\frac{\partial p}{\partial t} = -u \frac{\partial p}{\partial x} - v \frac{\partial p}{\partial y} - \gamma p \left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right) + \kappa \nabla^2 p$$

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| `gamma` | gamma | Ratio of specific heats (Cp/Cv) | 1.0 - 5/3 |
| `mu` | mu | Dynamic viscosity | 0.001 - 0.1 |
| `kappa` | kappa | Thermal/pressure diffusivity | 0.001 - 0.1 |

## Initial Condition Types

### acoustic-pulse (default)
Gaussian pressure perturbation on a uniform background. Produces expanding circular sound waves that demonstrate acoustic propagation.

### kelvin-helmholtz
Compressible shear layer with density stratification. A velocity shear profile (tanh) with a density jump across the interface seeds the Kelvin-Helmholtz instability, producing characteristic rolling vortex structures.

### density-blob
Gaussian density perturbation with isentropic pressure matching (p/p0 = (rho/rho0)^gamma). Shows expansion/compression dynamics as the blob equilibrates with the surrounding medium.

## References

- Anderson, J. D. (2003). *Modern Compressible Flow: With Historical Perspective*. McGraw-Hill.
- Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*. Springer.
- Landau, L. D., & Lifshitz, E. M. (1987). *Fluid Mechanics*. Pergamon Press.

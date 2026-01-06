# Burgers' Equation

The simplest nonlinear PDE combining advection and diffusion, serving as a fundamental model for shock wave formation and viscous flow dynamics.

## Description

Burgers' equation, studied extensively by Johannes Martinus Burgers in 1948, is the simplest equation that captures the essential physics of nonlinear wave steepening and shock formation. It serves as a one-dimensional model for understanding phenomena in gas dynamics, traffic flow, and fluid turbulence.

The equation exhibits two fundamentally different behaviors:

**Viscous case** ($\varepsilon > 0$): Diffusion competes with nonlinear steepening, producing smooth traveling waves and preventing true discontinuities. The viscous Burgers equation is exactly solvable via the Cole-Hopf transformation, which reduces it to the linear heat equation.

**Inviscid case** ($\varepsilon = 0$): Without diffusion, wave steepening proceeds unchecked. Characteristics can cross, leading to the formation of **shock waves** - discontinuous solutions that propagate at specific speeds determined by the Rankine-Hugoniot conditions.

Key phenomena:
- **Wave steepening**: Larger amplitudes travel faster, causing the wavefront to steepen
- **Shock formation**: In the inviscid limit, steepening leads to discontinuities
- **Shock thickness**: In the viscous case, $\varepsilon$ determines the shock width
- **Shock interaction**: Multiple shocks can overtake and merge

Burgers' equation is often the first nonlinear PDE encountered in applied mathematics courses and provides crucial intuition for more complex systems like the Navier-Stokes and Euler equations.

## Equations

**Viscous Burgers' equation**:

$$\frac{\partial u}{\partial t} = -u \frac{\partial u}{\partial x} + \varepsilon \frac{\partial^2 u}{\partial x^2}$$

**Inviscid Burgers' equation** ($\varepsilon = 0$):

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = 0$$

Where:
- $u(x,t)$ is the velocity/wave amplitude field
- $-u \, u_x$ is the nonlinear advection (self-transport)
- $\varepsilon$ is the viscosity/diffusion coefficient
- The term $u_x$ can be written as $u_{xb}$ (backward difference) in the numerical scheme

Conservation form:
$$\frac{\partial u}{\partial t} + \frac{\partial}{\partial x}\left(\frac{u^2}{2}\right) = \varepsilon \frac{\partial^2 u}{\partial x^2}$$

## Default Config

```yaml
solver: euler
dt: 0.04
dx: 2
domain_size: 1000
dimension: 1

boundary_x: neumann

parameters:
  epsilon: 0.05   # viscosity coefficient
```

## Parameter Variants

### BurgersEquation (Viscous)
Standard viscous Burgers equation:
- `epsilon = 0.05` (moderate viscosity)
- `dt = 0.04`, `dx = 2`
- `domain_size = 1000`
- Initial condition: Gaussian pulse at x = L/5
- 1D line plot visualization
- Shows smooth traveling wave with slight amplitude decay

### InviscidBurgers
Shock-capturing formulation using flux splitting:
- `epsilon = 0` (truly inviscid)
- Uses algebraic auxiliary variable for flux: $v = u^p$
- Reaction term: $-\frac{1}{p u^{p-2}} v_{xb}$ with $p=4$
- RK4 timestepping for better stability
- Initial condition: Gaussian pulse
- Shows shock formation and propagation

### InviscidBurgersShockInteraction
Multiple shock interaction demonstration:
- Same inviscid formulation
- Initial condition: Multiple Gaussian pulses of different heights
- Shows faster (taller) shocks overtaking slower ones
- Demonstrates shock merging dynamics

### Shock Formation Time

For initial condition $u(x,0) = u_0(x)$, shock forms at:
$$T_c = -\frac{1}{\min(u_0'(x))}$$

If $u_0'(x) < 0$ anywhere, a shock will form in finite time.

## References

- Burgers, J.M. (1948). "A mathematical model illustrating the theory of turbulence"
- Cole, J.D. (1951). "On a quasi-linear parabolic equation in aerodynamics"
- Hopf, E. (1950). "The partial differential equation $u_t + uu_x = \mu u_{xx}$"
- Whitham, G.B. (1974). "Linear and Nonlinear Waves" - Wiley

# Inviscid Burgers' Equation

The inviscid Burgers' equation is the simplest nonlinear hyperbolic PDE, serving as a prototype for understanding shock wave formation and weak solutions.

## Description

The inviscid Burgers' equation models nonlinear wave propagation without dissipation. Unlike the viscous version, solutions develop discontinuities (shocks) in finite time from smooth initial data. This makes it a fundamental test case for numerical methods handling discontinuities.

### Key Physical Behaviors

**Wave steepening**: In the inviscid limit, larger amplitudes travel faster than smaller ones. This causes wave fronts to steepen over time.

**Shock formation**: Eventually, the wave front becomes vertical and a true discontinuity (shock) forms. This happens in finite time.

**Shock merging**: When a faster shock overtakes a slower one, they merge into a single shock. The merged shock propagates at a speed determined by the Rankine-Hugoniot jump condition.

**Entropy condition**: Multiple weak solutions exist mathematically. The physically relevant (entropy) solution is the one obtained as the limit of vanishing viscosity.

### Applications

- **Gas dynamics**: Simplified model for shock waves in compressible flow
- **Traffic flow**: Cars bunch up behind slow vehicles
- **Shallow water**: Wave breaking and bores
- **Numerical methods**: Test case for shock-capturing schemes

## Equations

Conservation form:
$$\frac{\partial u}{\partial t} + \frac{\partial}{\partial x}\left(\frac{u^2}{2}\right) = 0$$

Non-conservative form:
$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = 0$$

The characteristic velocity is $u$ itself, explaining why larger $u$ travels faster.

### Shock Speed

At a shock, the Rankine-Hugoniot condition gives:
$$s = \frac{u_L + u_R}{2}$$

where $s$ is the shock speed, $u_L$ is the value to the left, and $u_R$ is the value to the right.

## Default Config

```yaml
preset: inviscid-burgers
solver: rk4
dt: 0.03
domain_size: 200

bc:
  x: neumann
  y: neumann

parameters:
  epsilon: 0.0  # Truly inviscid (set > 0 for regularization)

init:
  type: default
  params:
    offset: 0.001
    amplitude: 0.1
```

## Parameter Variants

### Truly Inviscid (epsilon = 0)
- Shocks form with zero width
- Requires careful numerics
- May develop oscillations (Gibbs phenomenon)

### Regularized (epsilon = 1e-4)
- Shocks have finite width proportional to epsilon
- More stable numerically
- Approaches inviscid solution as epsilon -> 0

## Initial Condition Variants

### Single Pulse (default)
Initial condition: $u = 0.001 + 0.1 \exp(-0.00005(x-L_x/5)^2)$

A single Gaussian pulse that steepens into a shock. The small offset (0.001) prevents numerical issues when $u \to 0$.

### Shock Interaction
Multiple Gaussians at different positions and heights:
- Taller shocks travel faster
- Shocks merge when they collide
- Demonstrates conservation of momentum

## Numerical Considerations

The inviscid Burgers' equation is challenging numerically:

1. **Timestep restriction**: Use CFL-based timestep (dt < dx/max|u|)
2. **Shock capturing**: Standard centered differences create oscillations near shocks
3. **Regularization**: Adding tiny viscosity (epsilon ~ 1e-4) helps stability

Visual PDE uses a flux-splitting approach with an algebraic auxiliary variable for improved shock capturing. This formulation writes the flux as $v = u^p$ and uses backward differences for the derivative.

## References

- Burgers, J. M. (1948). A mathematical model illustrating the theory of turbulence. Advances in Applied Mechanics, 1, 171-199.
- Lax, P. D. (1957). Hyperbolic systems of conservation laws II. Communications on Pure and Applied Mathematics, 10(4), 537-566.
- [Inviscid Burgers' equation - Wikipedia](https://en.wikipedia.org/wiki/Burgers'_equation#Inviscid_Burgers'_equation)
- VisualPDE Inviscid Burgers preset

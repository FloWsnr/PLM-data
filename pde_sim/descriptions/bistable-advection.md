# Bistable Advection

The bistable advection equation combines the Allen-Cahn bistable reaction with directed flow, modeling population invasion in flowing environments like rivers or ocean currents.

## Description

This equation extends the classic bistable Allen-Cahn equation by adding an advection term representing flow. The bistable reaction term models a population with an Allee effect - a minimum population density threshold below which the population cannot sustain itself.

Applications include:

- **River ecology**: Populations of fish or invertebrates in flowing water
- **Ocean currents**: Plankton or larvae dispersal with oceanic transport
- **Atmospheric transport**: Spread of airborne organisms or pollutants
- **Microfluidics**: Pattern formation in flowing chemical systems
- **Invasion biology**: How environmental flows affect species invasions

### Key Physical Behaviors

**Invasion fronts**: The bistable reaction creates traveling wave fronts that separate regions of high population density from empty regions. The front speed depends on the Allee threshold parameter $a$.

**Flow-front interaction**: The advection term can either assist or oppose the invasion front depending on the flow direction. When flow opposes invasion, it can halt or reverse the front. When flow assists, invasion accelerates.

**Critical threshold**: At $a = 0.5$, the front is stationary without flow. Values $a < 0.5$ favor invasion while $a > 0.5$ favor extinction.

## Equations

$$\frac{\partial u}{\partial t} = D \nabla^2 u + u(u-a)(1-u) + V \left(\cos\theta \frac{\partial u}{\partial x} + \sin\theta \frac{\partial u}{\partial y}\right)$$

where:
- $u(x, y, t)$ is the population density (between 0 and 1)
- $D$ is the diffusion coefficient
- $a$ is the Allee threshold (unstable equilibrium)
- $V$ is the flow velocity magnitude
- $\theta$ is the flow direction in radians

The reaction term $u(u-a)(1-u)$ has three equilibria:
- $u = 0$ (stable, extinction)
- $u = a$ (unstable, Allee threshold)
- $u = 1$ (stable, carrying capacity)

## Default Config

```yaml
preset: bistable-advection
solver: euler
dt: 0.005
domain_size: 100

bc:
  x: neumann
  y: neumann

parameters:
  D: 0.02      # diffusion coefficient
  a: 0.48      # Allee threshold (near critical)
  theta: 1.8   # flow direction (radians)
  V: 0.5       # flow velocity

init:
  type: step
  params:
    direction: x
    step_position: 0.3
    value_low: 0.0
    value_high: 1.0
    smooth: true
```

## Parameter Variants

### No Flow (V = 0)
- Standard bistable Allen-Cahn equation
- Front propagates at intrinsic speed determined by $a$

### Flow Opposing Invasion
- $\theta$ chosen so flow points against invasion direction
- Can slow, halt, or reverse invasion front

### Flow Assisting Invasion
- $\theta$ chosen so flow points with invasion direction
- Accelerates invasion front

### Critical Threshold (a = 0.5)
- Without flow: stationary front
- With flow: front moves purely due to advection

## References

- [Allen-Cahn equation - Wikipedia](https://en.wikipedia.org/wiki/Allen%E2%80%93Cahn_equation)
- [Allee effect - Wikipedia](https://en.wikipedia.org/wiki/Allee_effect)
- VisualPDE BistableAdvection preset

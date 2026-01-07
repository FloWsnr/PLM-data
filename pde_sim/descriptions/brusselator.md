# Brusselator Model

A two-component reaction-diffusion system modeling oscillating chemical reactions, named after Brussels where it was developed by Ilya Prigogine's group.

## Description

The Brusselator is a theoretical model developed in Brussels by Ilya Prigogine and colleagues to study the emergence of dissipative structures in chemical systems far from equilibrium. It was designed to demonstrate how self-organization and pattern formation can arise from simple chemical kinetics, a concept central to Prigogine's Nobel Prize-winning work on non-equilibrium thermodynamics.

The model captures the essential features of oscillating chemical reactions like the Belousov-Zhabotinsky reaction. It describes the interaction between two chemical species (morphogens): an activator and an inhibitor that react and diffuse with different rates.

Key behaviors include:
- **Turing patterns**: Stable spatial patterns (spots, stripes) when D exceeds a critical threshold
- **Oscillations**: Periodic temporal oscillations in concentrations
- **Turing-Wave instabilities**: In hyperbolic extensions, combined spatial and temporal instabilities

The Brusselator revealed fundamental principles of pattern formation: the coupling between nonlinear reaction kinetics and differential diffusion rates, where the inhibitor must diffuse more rapidly than the activator. This activator-inhibitor principle has become universal for explaining pattern formation in chemical, ecological, physical, and biological systems.

### Stability Conditions
- Homogeneous equilibrium is stable for $b - 1 < a^2$
- Turing instability occurs for $D > \frac{a^2}{(\sqrt{b}-1)^2}$
- For $a = 2$, $b = 3$: critical $D_c \approx 7.46$

## Equations

$$\frac{\partial u}{\partial t} = \nabla^2 u + a - (b+1)u + u^2 v$$

$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + bu - u^2 v$$

Where:
- $u$ is the activator concentration
- $v$ is the inhibitor concentration
- $a, b > 0$ are kinetic parameters
- $D_v$ (`Dv`) is the diffusion coefficient ratio

### Hyperbolic Extension
$$\tau \frac{\partial^2 u}{\partial t^2} + \frac{\partial u}{\partial t} = D \nabla^2 u + a - (b+1)u + u^2 v$$

$$\tau \frac{\partial^2 v}{\partial t^2} + \frac{\partial v}{\partial t} = \nabla^2 v + bu - u^2 v$$

## Default Config

```yaml
solver: euler
dt: 0.001
dx: 1.0
domain_size: 100

boundary_x: periodic
boundary_y: periodic

parameters:
  Dv: 8    # range: [7, 9], step: 0.1
  a: 2
  b: 3
```

## Parameter Variants

### brusselator (Standard)
Standard configuration near the Turing instability threshold.
- `Dv = 8` (range: [7, 9]): Just above critical threshold (~7.46)
- `a = 2`, `b = 3`: Classic parameter values
- Initial conditions: `u(0) = a`, `v(0) = b/a`

### Hyperbolic Variants (Turing-Wave)
Extensions with second-order time derivatives enabling Turing-Wave instabilities:
- `tau > 0`: Inertial parameter
- `D < 1`: Diffusion ratio on u equation (reversed from standard)
- Produces oscillating spatial patterns and wave-like spatiotemporal behavior

## References

- Prigogine, I., & Lefever, R. (1968). Symmetry breaking instabilities in dissipative systems. II. Journal of Chemical Physics, 48(4), 1695-1700.
- Nicolis, G., & Prigogine, I. (1977). Self-Organization in Nonequilibrium Systems. Wiley.
- Turing, A. M. (1952). The chemical basis of morphogenesis. Philosophical Transactions of the Royal Society B, 237(641), 37-72.
